import os

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path, encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):        # relaions.dict和entities.dict中的id都是按顺序排列的
            rel, id = line.strip().split("\t")
            index[rel] = id
            rev_index[id] = rel
    return index, rev_index

entity2id, id2entity = load_index(os.path.join('entity2id.txt'))
relation2id, id2relation = load_index(os.path.join('relation2id.txt'))

count = 0
count1 = 0
word_list = set()
for entity_str in entity2id.keys():
    if "(" in entity_str and ")" in entity_str:
        count += 1
        begin = entity_str.find('(')
        end = entity_str.find(')')
        w1 = entity_str[:begin].strip()
        w2 = entity_str[begin+1: end].strip()
        if w2 not in entity2id.keys():
            count1 += 1
        word_list.add(w1)
        word_list.add(w2)
    else:
        word_list.add(entity_str)

num_word = len(word_list)
word2id={}
k = 0
for id, word in enumerate(word_list):
    if word in entity2id.keys():
        word2id[word] = entity2id[word]
    else:
        word2id[word] = len(entity2id.keys()) + k
        k=k+1
print(k)#最后加了一次

# id2word = {id: word for id, word in enumerate(word_list)}
# print(word2id)
# print(id2word)

print(len(word2id.keys()))
print(float(count)/len(entity2id.keys()))
print(float(count1)/float(count))

with open("word2id.txt", "w", encoding='utf-8') as f:
    for word in word2id.keys():
        f.write(word + "\t" + str(word2id[word])+'\n')

eid2wid = []
for id in range(len(id2entity.keys())):
    entity_str = id2entity[str(id)]
    if "(" in entity_str and ")" in entity_str:
        count += 1
        begin = entity_str.find('(')
        end = entity_str.find(')')
        w1 = entity_str[:begin].strip()
        w2 = entity_str[begin+1: end].strip()
        eid2wid.append([str(entity2id[entity_str]), "0", str(word2id[w1])])  # isA关系
        eid2wid.append([str(entity2id[entity_str]), "1", str(word2id[w2])])  # 隶属关系
    else:
        eid2wid.append([str(entity2id[entity_str]), "2", str(word2id[entity_str])])

#使用LLM，为每个实体添加属性
# 整合entiy2id 和 word2id  得到包含word 的new_entiy2id
def merge_dicts(entity2id, word2id):
    """ 合并两个字典，保留 entity2id 的优先级 """
    new_entity2id = entity2id.copy()  # 先复制 entity2id
    new_entity2id.update({k: v for k, v in word2id.items() if k not in new_entity2id})  # 仅添加不重复的键
    return new_entity2id

new_entity2id = merge_dicts(entity2id, word2id)
print('new_entity2id', len(new_entity2id))

# 调用llm 对new_entiy2id处理

import os
import re
import json
import math
from openai import OpenAI
#
# ##############################
# # 配置参数
# ##############################
client = OpenAI(
    api_key="xx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL_NAME ="qwq-plus"
#
# # 预定义角色列表（共12个角色，对应编码 1～12）
ROLE_DICT = {
    "Countries/Regions": "1", #国家与地区
    "Government/Political Organizations": "2",#政府机构 & 政党组织
    "Political Figures/Leaders": "3", #政治人物 & 领导人
    "Law Enforcement/Military": "4", #执法 & 军事机构
    "Crime/Armed Groups": "5", #犯罪 & 武装组织
    "Economy/Business": "6", #经济 & 贸易组织
    "Judiciary/Legal Professionals": "7",#司法 & 法律人员
    "Social Groups/Citizens": "8",#社会群体 & 公民身份
    "Media/Press": "9",#媒体 & 新闻机构
    "Education/Research": "10",#教育 & 研究机构
    "Healthcare/Medical": "11",#卫生 & 医疗
    "Religion/Culture": "12", #1宗教 & 文化
    "Technology/Information":"13",# 科技 & 信息
    "International Organizations/Alliances":"14",  # 国际组织 & 联盟 (
    "Local Governments/Municipalities":"15",#地方政府 & 市政管理
    "Unknown": "16",  # 7338
}

# ##############################
# # 辅助函数：提取 JSON 块
# ##############################
#
def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group()  # 提取匹配的 JSON 字符串
        data = json.loads(json_str)  # 转换为 Python 字典
        # print('data',data)  # 输出解析后的字典
    else:
        print("未找到 JSON 数据")

    # w_json ={}
    # for k,v in data.items():
    #     try:
    #         w_json[k] = ROLE_DICT[v]
    #     except Exception as e:
    return data

##############################
# LLM API 调用函数（返回角色归类）
##############################
def llm_get_relation(w,n):
    """
    对输入的两部分进行角色归类，如果只对 w1 归类，则 w2 传空字符串。
    返回 JSON 格式：{"primary_role": "<角色名称>", "secondary_role": "<角色名称>"}
    """
    prompt = (
        f"下面是一组预定义的角色列表：\n"
        f"1.Countries/Regions\n"
        f"2.Government/Political Organizations\n"
        f"3.Political Figures/Leaders\n"
        f"4.Law Enforcement/Military\n"
        f"5.Crime/Armed Groups\n"
        f"6.Economy/Business\n"
        f"7.Judiciary/Legal Professionals\n"
        f"8.Social Groups/Citizens\n"
        f"9.Media/Press\n"
        f"10.Education/Research\n"
        f"11.Healthcare/Medical\n"
        f"12.Religion/Culture\n"
        f"13.Technology/Information\n"
        f"14.International Organizations/Alliances\n"
        f"15.Local Governments/Municipalities\n"
        f"16.Unknown\n"
        f"输入的实体名称为：{w}，一共包含{n}个实体：，这些实体名称全来至ICEWS（综合危机预警系统），旨在通过新闻数据分析，提供全球政治和社会事件预警。\n"
        f"输入的实体名称可能有多个，请为每个实体选择对应的角色。要求：1.每个实体只分配一个角色分类；2.严格保持保持输入实体名称和输出实体名称一致；3.严格保持输出实体顺序与输入顺序一致。4.严格保持输入的实体数量与输出的结果数量一致。\n"
        f"输出JSON格式结果，格式如下：\n"
        "{\"China\":\"1\", \"Citizen_(Nigeria)\":\"8\", \"Government (South Korea)\":\"2\"}\n"
    )

    completion = client.chat.completions.create(
        model="qwq-plus",  # 此处以 qwq-32b 为例，可按需更换模型名称
        messages=[
            {"role": "system", "content": "You are a factual document generator following strict guidelines."},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        max_tokens=1024,
        # }
    )
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                # print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content
    json_text = extract_json(answer_content)
    return json_text


use_llm = False
only_llm = True
segment_count = 40  # 分成 10 片段
patch = 50
start_segment = 0  #从哪个开始

if use_llm:
    eid2wid_attr = {f'eid2wid_attr_{j}': [] for j in range(segment_count)}  # 存储 10 个片段
    new_entity_list = list(new_entity2id.keys())  # 将字典转换为有序列表
    chunk_size = math.ceil(len(new_entity_list) / segment_count)
    print(len(new_entity_list), chunk_size)

    #定义一个超参数，可以从第一个片段开始重新处理
    for segment in range(start_segment, segment_count):
        start_idx = segment * chunk_size
        end_idx = (segment + 1) * chunk_size if segment < segment_count - 1 else len(new_entity_list)
        entity_chunk = new_entity_list[start_idx:end_idx]  # 获取当前片段的实体列表
        i = 0
        entity_batch = []
        global_entity_idx = start_idx  # 全局实体 ID，从当前 segment 的起始索引开始
        for k in entity_chunk:##############定义从那个片段开始处理
            entity_batch.append(k)
            i += 1
            # 每 50 个或最后一个处理一次
            if i == patch or k == entity_chunk[-1]:
                print(f"Processing segment {segment}, batch: {str(entity_batch)}")
                # 调用 LLM 解析，确保返回结果长度符合预期
                h = 0
                while True:
                    num_s = len(entity_batch)
                    for o in range(5):
                        try:
                            json_text = llm_get_relation(str(entity_batch), num_s)
                            break
                        except Exception as e:
                            print(f"LLM 调用出错: {e}")
                            json_text = None

                    # If all retries failed, split the batch
                    if json_text is None:
                        print("All retries failed, splitting batch...")
                        split_size = 10
                        all_sub_results = {}
                        for i in range(0, len(entity_batch), split_size):
                            sub_batch = entity_batch[i:i + split_size]
                            try:
                                sub_result = llm_get_relation(str(sub_batch), len(sub_batch))
                            except Exception as e:
                                print(f"Sub-batch failed: {e}")
                                sub_result = {entity: "-1" for entity in sub_batch}
                            all_sub_results.update(sub_result)
                        json_text = all_sub_results

                    if len(json_text) == patch:
                        break
                    elif k == entity_chunk[-1]:
                        break
                    h = h +1
                    missing_entities = [entity for entity in entity_batch if entity not in json_text]
                    print(f"{segment}, batch size mismatch. Missing entities in response: {missing_entities}")
                    if h > 2 and len(missing_entities) <3:
                        complete_json = {}
                        for entity in entity_batch:
                            complete_json[entity] = json_text.get(entity, "-1")
                        json_text = complete_json
                        break

                entity_batch = []
                i = 0
                print(f'json_text (segment {segment}):', json_text)
                # 重新整合 eid2wid_attr_f'{segment}'

                for k_j, v_j in json_text.items():
                    try:#
                        ent = str(entity2id[k_j])######实体名写错
                        # print(ent,str(global_entity_idx))
                        if ent != str(global_entity_idx):
                            raise ValueError("ent != str(global_entity_idx)")
                    except Exception as e:
                        ent = str(global_entity_idx)
                        print(f"整体：LLM返回实体名称与输入名称不一致 (segment {segment}): {e}")
                    global_entity_idx += 1  # 更新全局索引
                    if v_j == '-1':#处理LLM出错的情况
                        for o in range(3):
                            try:
                                k_json = llm_get_relation(k_j, 1)
                                if len(k_json) != 1:
                                    raise ValueError("k_json length is not 1")
                                v = next(iter(k_json.values()))
                                eid2wid_attr[f'eid2wid_attr_{segment}'].append([ent, "3", str(int(v) + len(new_entity2id) - 1)])
                                break
                            except Exception as e:
                                eid2wid_attr[f'eid2wid_attr_{segment}'].append([ent, "3", str(len(new_entity2id) + len(ROLE_DICT.keys()) - 1)])  # 12未知
                                print(f"LLM返回结果：{k_json}")
                                print(f"单个：LLM返回实体名称与输入名称不一致 (segment {segment}): {e}")
                    else:
                        eid2wid_attr[f'eid2wid_attr_{segment}'].append([ent, "3", str(int(v_j) + len(new_entity2id) - 1)])#正常
            # print(eid2wid_attr)
        # **存储该片段的结果**
        with open(f"e-w-graph-llm-{segment}.txt", "w", encoding='utf-8') as f:
            for line in eid2wid_attr[f'eid2wid_attr_{segment}']:
                f.write("\t".join(line) + '\n')

if only_llm:
    eid2wid =[]
else:
    pass
# **为 12 个类别属性增加类别节点**
for k, v in ROLE_DICT.items():
    eid2wid.append([str(int(v) + len(new_entity2id) - 1), "0", str(len(new_entity2id) + len(ROLE_DICT.keys()))])  # 定义为类别节点

# **整合所有 eid2wid_attr_f'{i}'**
# 从新读取e-w-graph-llm-{segment}.txt文件，再整合
for segment in range(segment_count):
    filename = f"e-w-graph-llm-{segment}.txt"
    if os.path.exists(filename):  # 确保文件存在
        with open(filename, "r", encoding='utf-8') as f:
            for line in f:
                #关系编号修改为1
                llm_list = line.strip().split("\t")
                llm_list[1] = str(int(llm_list[1])-2)
                eid2wid.append(llm_list)  # 读取并转换为列表

if only_llm:
    txt_name = "e-w-graph-llm-only.txt"
else:
    txt_name = "e-w-graph-llm.txt"

# **最终保存完整数据**
with open(txt_name, "w", encoding='utf-8') as f:
    for line in eid2wid:
        f.write("\t".join(line) + '\n')

print("数据处理完成，所有数据已保存！")






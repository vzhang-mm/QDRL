import argparse
def parse_opts():
    parser = argparse.ArgumentParser('Action Recognition')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default="GDELT",
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    # parser.add_argument("--multi-step", action='store_true', default=False,
    #                     help="do multi-steps inference without ground truth")
    # parser.add_argument("--topk", type=int, default=10,
    #                     help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph", action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-his-graph", action='store_true', default=False,
                        help="use the info of his graph")
    # parser.add_argument("--add-rel-word", action='store_true', default=False,
    #                     help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--pre-type", type=str, default="short",
                        help=["GRU", "TF"])
    parser.add_argument("--q-type", type=str, default="short",
                    help=["conv", "liner"])
    parser.add_argument("--add_query", action='store_true', default=False,
                    help="")
    parser.add_argument("--use_onehot", action='store_true', default=False,
                    help="")

    # parser.add_argument("--use-cl", action='store_true', default=False,
    #                     help="use the info of  contrastive learning")
    # parser.add_argument("--temperature", type=float, default=0.07,######################对比学习部分，可以删
    #                     help="the temperature of cl")
    # configuration for encoder RGCN stat
    # parser.add_argument("--weight", type=float, default=1,
    #                     help="weight of static constraint")
    parser.add_argument("--pre-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    # parser.add_argument("--discount", type=float, default=1,
    #                     help="discount of weight of static constraint")
    # parser.add_argument("--angle", type=int, default=10,
    #                     help="evolution speed")
    parser.add_argument("--encoder", type=str, default="uvrgcn",  # {uvrgcn,kbat,compgcn}
                        help="method of encoder")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")
    # parser.add_argument("--aggregation", type=str, default="none",
    #                     help="method of aggregation")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--n-bases", type=int, default=100,##128
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    # parser.add_argument("--entity-prediction", action='store_true', default=True,
    #                     help="add entity prediction loss")
    # parser.add_argument("--split_by_relation", action='store_true', default=False,
    #                     help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    # parser.add_argument("--input-dropout", type=float, default=0.2,
    #                     help="input dropout for decoder ")
    # parser.add_argument("--hidden-dropout", type=float, default=0.2,
    #                     help="hidden dropout for decoder")
    # parser.add_argument("--feat-dropout", type=float, default=0.2,
    #                     help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    # parser.add_argument("--test-history-len", type=int, default=20,
    #                     help="history length for test")
    # parser.add_argument("--dilate-len", type=int, default=1,
    #                     help="dilate history graph")

    args = parser.parse_args()

    return args
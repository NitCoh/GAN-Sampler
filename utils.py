
def handle_args_for_train(args):
    """
    Return the args as tuple
    :param args:
    :return:
    """
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dim = args.dim
    optim = args.optimizer
    return batch_size, epochs, lr, dim, optim
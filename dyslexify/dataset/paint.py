from dyslexify.dataset import RTA100


class Paint(RTA100):

    pass


if __name__ == "__main__":
    dataset = Paint(root="/datasets/paint_ds")
    print(dataset[0])
    import code

    code.interact(local=dict(globals(), **locals()))

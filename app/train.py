import openLLV as llv


def run_train(model, root_dir, epochs, batch_size, device):
    return llv.train(
        model,
        root_dir=root_dir,
        epochs=int(epochs),
        batch_size=int(batch_size),
        device=device,
    )

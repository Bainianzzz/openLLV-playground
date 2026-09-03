import openLLV as llv


def run_train(model, root_dir, epochs, batch_size, device, save_dir):
    kwargs = {}
    if save_dir and save_dir.strip():
        kwargs["output_dir"] = save_dir.strip()
    return llv.train(
        model,
        root_dir=root_dir,
        epochs=int(epochs),
        batch_size=int(batch_size),
        device=device,
        **kwargs,
    )

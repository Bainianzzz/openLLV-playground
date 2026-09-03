import openLLV as llv


def run_evaluate(en_dir, ref_dir, metrics, save_dir):
    """评估图像质量；save_dir 非空时指定结果保存文件。"""
    ref = ref_dir.strip() if ref_dir and ref_dir.strip() else None
    save_path = save_dir.strip() if save_dir and save_dir.strip() else None
    return llv.evaluate(
        en_img_dir=en_dir,
        ref_img_dir=ref,
        metrics=metrics,
        save_path=save_path,
    )

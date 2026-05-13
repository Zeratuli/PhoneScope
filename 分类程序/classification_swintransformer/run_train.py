import config as cfg
from datasets import build_datasets
from trainer import train


def main():
    train_ds, val_ds, test_ds = build_datasets(
        cfg.DATA_DIR, cfg.IMG_SIZE,
        aug_level=cfg.AUG_LEVEL,
        read_backend=cfg.READ_BACKEND,
        cache_mode=cfg.CACHE_MODE
    )

    train(cfg, train_ds, val_ds, test_ds)


if __name__ == "__main__":
    main()

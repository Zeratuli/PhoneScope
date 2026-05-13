from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.resolve()


def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    print(f"[OK] dir created: {p}")


def write_file(p: Path, content: str):
    if p.exists():
        print(f"[SKIP] file exists: {p}")
        return
    p.write_text(content, encoding="utf-8")
    print(f"[OK] file created: {p}")


def main():
    print(f"[INFO] Initializing Swin classification project at:\n{ROOT}\n")

    # ===============================
    # 1. 创建目录结构
    # ===============================
    dirs = [
        ROOT / "data" / "train",
        ROOT / "data" / "val",
        ROOT / "data" / "test",
        ROOT / "runs" / "swin",
        ROOT / "utils",
    ]

    for d in dirs:
        mkdir(d)

    # ===============================
    # 2. train_swin.py
    # ===============================
    train_swin_py = (
        "import os\n"
        "import time\n"
        "import argparse\n"
        "from pathlib import Path\n\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.data import DataLoader\n"
        "from torchvision import datasets, transforms\n\n"
        "import timm\n"
        "from tqdm import tqdm\n\n"
        "def build_transforms(img_size=224):\n"
        "    train_tf = transforms.Compose([\n"
        "        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),\n"
        "        transforms.RandomHorizontalFlip(),\n"
        "        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),\n"
        "        transforms.ToTensor(),\n"
        "        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),\n"
        "    ])\n\n"
        "    val_tf = transforms.Compose([\n"
        "        transforms.Resize(int(img_size * 1.15)),\n"
        "        transforms.CenterCrop(img_size),\n"
        "        transforms.ToTensor(),\n"
        "        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),\n"
        "    ])\n"
        "    return train_tf, val_tf\n\n"
        "@torch.no_grad()\n"
        "def evaluate(model, loader, device):\n"
        "    model.eval()\n"
        "    correct, total, loss_sum = 0, 0, 0.0\n"
        "    criterion = nn.CrossEntropyLoss()\n"
        "    for x, y in loader:\n"
        "        x, y = x.to(device), y.to(device)\n"
        "        logits = model(x)\n"
        "        loss = criterion(logits, y)\n"
        "        loss_sum += loss.item() * x.size(0)\n"
        "        correct += (logits.argmax(1) == y).sum().item()\n"
        "        total += y.size(0)\n"
        "    return loss_sum / total, correct / total\n\n"
        "def main(args):\n"
        "    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'\n"
        "    print(f'[INFO] device = {device}')\n\n"
        "    train_dir = os.path.join(args.data_dir, 'train')\n"
        "    val_dir = os.path.join(args.data_dir, 'val')\n\n"
        "    train_tf, val_tf = build_transforms(args.img_size)\n"
        "    train_ds = datasets.ImageFolder(train_dir, train_tf)\n"
        "    val_ds = datasets.ImageFolder(val_dir, val_tf)\n\n"
        "    num_classes = len(train_ds.classes)\n"
        "    print(f'[INFO] classes ({num_classes}): {train_ds.classes}')\n\n"
        "    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)\n"
        "    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)\n\n"
        "    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes).to(device)\n"
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)\n"
        "    criterion = nn.CrossEntropyLoss()\n\n"
        "    out_dir = Path(args.out_dir)\n"
        "    out_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "    best_acc = 0.0\n"
        "    for epoch in range(1, args.epochs + 1):\n"
        "        model.train()\n"
        "        for x, y in train_loader:\n"
        "            x, y = x.to(device), y.to(device)\n"
        "            optimizer.zero_grad()\n"
        "            logits = model(x)\n"
        "            loss = criterion(logits, y)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n\n"
        "        val_loss, val_acc = evaluate(model, val_loader, device)\n"
        "        print(f'[VAL] loss={val_loss:.4f}, acc={val_acc:.4f}')\n\n"
        "        torch.save(model.state_dict(), out_dir / 'last.pt')\n"
        "        if val_acc > best_acc:\n"
        "            best_acc = val_acc\n"
        "            torch.save(model.state_dict(), out_dir / 'best.pt')\n\n"
        "    print(f'[DONE] Best Acc = {best_acc:.4f}')\n\n"
        "if __name__ == '__main__':\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--data_dir', default='data')\n"
        "    parser.add_argument('--out_dir', default='runs/swin')\n"
        "    parser.add_argument('--model', default='swin_tiny_patch4_window7_224')\n"
        "    parser.add_argument('--epochs', type=int, default=30)\n"
        "    parser.add_argument('--batch_size', type=int, default=32)\n"
        "    parser.add_argument('--lr', type=float, default=3e-4)\n"
        "    parser.add_argument('--wd', type=float, default=0.05)\n"
        "    parser.add_argument('--workers', type=int, default=4)\n"
        "    args = parser.parse_args()\n"
        "    main(args)\n"
    )

    write_file(ROOT / "train_swin.py", train_swin_py)

    # ===============================
    # 3. README.md
    # ===============================
    readme = (
        "# Swin Transformer 手机型号分类\n\n"
        "## 目录结构\n\n"
        "classification/\n"
        "├─ data/\n"
        "│  ├─ train/\n"
        "│  ├─ val/\n"
        "│  └─ test/\n"
        "├─ runs/\n"
        "│  └─ swin/\n"
        "├─ train_swin.py\n"
        "└─ init_project.py\n\n"
        "## 训练方式\n\n"
        "python train_swin.py --data_dir data --epochs 30 --batch_size 32\n"
    )

    write_file(ROOT / "README.md", readme)

    # ===============================
    # 4. .gitignore
    # ===============================
    gitignore = (
        "__pycache__/\n"
        "*.pt\n"
        "runs/\n"
        "data/\n"
    )

    write_file(ROOT / ".gitignore", gitignore)

    print("\n[ALL DONE] Project initialized successfully 🎉")


if __name__ == "__main__":
    main()

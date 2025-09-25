# scripts/download_vggface2hq.py
from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile

REPO = "ProgramComputer/VGGFace2-HQ"
TRAIN_ZIPS = [f"train/VGGFac{str(i).zfill(2)}.zip" for i in range(1,11)]
TEST_ZIP = "test/test.zip"

out_root = Path("data/vggface2hq"); out_root.mkdir(parents=True, exist_ok=True)

def fetch_and_extract(filename, dest):
    fp = hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset")
    with zipfile.ZipFile(fp) as z:
        z.extractall(dest)

# Train
train_dir = out_root / "train"; train_dir.mkdir(exist_ok=True)
for z in TRAIN_ZIPS:
    print(">>", z); fetch_and_extract(z, train_dir)

# Test
test_dir = out_root / "test"; test_dir.mkdir(exist_ok=True)
print(">>", TEST_ZIP); fetch_and_extract(TEST_ZIP, test_dir)

print("Done. Estructura en", out_root.resolve())


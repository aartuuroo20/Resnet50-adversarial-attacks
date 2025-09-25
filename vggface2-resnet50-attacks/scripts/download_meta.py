from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

# Repo y archivo
REPO = "ProgramComputer/VGGFace2"
FILENAME = "meta/identity_meta.csv"

# Carpeta destino en tu proyecto
out_path = Path("data/vggface2hq/meta")
out_path.mkdir(parents=True, exist_ok=True)

# Descarga temporal con huggingface_hub
csv_file = hf_hub_download(repo_id=REPO, filename=FILENAME, repo_type="dataset")

# Copia al destino
shutil.copy(csv_file, out_path / "identity_meta.csv")
print("Descargado a:", out_path / "identity_meta.csv")

import os
import shutil
import pathspec

# Configuration
SOURCE_ROOT = os.getcwd()
DEST_ROOT = os.path.join(SOURCE_ROOT, "docs_collection")
# Always ignore these to prevent recursion or system pollution, regardless of gitignore
CRITICAL_IGNORES = {".git", "site", "docs_collection", ".vscode", "__pycache__"}

ALLOWED_EXTENSIONS = {".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

def load_gitignore_spec():
    gitignore_path = os.path.join(SOURCE_ROOT, ".gitignore")
    lines = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    return pathspec.PathSpec.from_lines('gitwildmatch', lines)

def main():
    # Clean previous build
    if os.path.exists(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)
    os.makedirs(DEST_ROOT)

    spec = load_gitignore_spec()
    print(f"Scanning {SOURCE_ROOT} (filtering via .gitignore)...")

    # Copy Custom Static Assets (e.g. KaTeX config)
    overrides_dir = os.path.join(SOURCE_ROOT, "overrides")
    if os.path.exists(overrides_dir):
        print(f"Copying overrides from {overrides_dir}...")
        for root, dirs, files in os.walk(overrides_dir):
            rel = os.path.relpath(root, overrides_dir)
            dest = os.path.join(DEST_ROOT, rel)
            if not os.path.exists(dest):
                os.makedirs(dest)
            for f in files:
                shutil.copy2(os.path.join(root, f), os.path.join(dest, f))

    for dirpath, dirnames, filenames in os.walk(SOURCE_ROOT):
        rel_dir = os.path.relpath(dirpath, SOURCE_ROOT)
        if rel_dir == ".":
            rel_dir = ""

        # Filter directories in-place
        # 1. Critical ignores
        # 2. Gitignore matches
        valid_dirs = []
        for d in dirnames:
            if d in CRITICAL_IGNORES:
                continue
            
            child_rel = os.path.join(rel_dir, d) if rel_dir else d
            
            # Check if directory matches gitignore. 
            # Adding trailing slash helps match directory-specific rules (like "foo/")
            if spec.match_file(child_rel) or spec.match_file(child_rel + "/"):
                continue
            valid_dirs.append(d)
        
        dirnames[:] = valid_dirs

        dest_dir = os.path.join(DEST_ROOT, rel_dir) if rel_dir else DEST_ROOT

        for file in filenames:
            file_rel = os.path.join(rel_dir, file) if rel_dir else file

            if spec.match_file(file_rel):
                continue

            src_file = os.path.join(dirpath, file)
            _, ext = os.path.splitext(file)
            ext = ext.lower()
            
            should_copy = False
            target_name = file

            # 1. Handle Markdown files
            if ext in ALLOWED_EXTENSIONS:
                should_copy = True
                # If it's the root README, rename to index.md for the Homepage
                if rel_dir == "" and file.lower() == "readme.md":
                    target_name = "index.md"
            
            # 2. Handle Images (strictly inside 'imgs' folders as requested)
            elif os.path.basename(dirpath) == "imgs" and ext in IMAGE_EXTENSIONS:
                should_copy = True

            if should_copy:
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                dest_file = os.path.join(dest_dir, target_name)
                
                try:
                    # Just copy files directly, no processing
                    shutil.copy2(src_file, dest_file)
                except Exception as e:
                    print(f"Error copying {src_file}: {e}")

    print(f"Success! Content prepared in '{DEST_ROOT}' using .gitignore rules.")

if __name__ == "__main__":
    main()

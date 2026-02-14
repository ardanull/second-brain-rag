import os
import sys
import mimetypes
import httpx

def main():
    if len(sys.argv) < 2:
        print("usage: python import_folder.py <folder> [api_base]")
        return 2
    folder = sys.argv[1]
    api = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000").rstrip("/")
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            p = os.path.join(root, n)
            if n.lower().endswith((".pdf", ".md", ".txt")):
                files.append(p)
    files.sort()
    if not files:
        print("no supported files")
        return 1
    with httpx.Client(timeout=180.0) as client:
        for p in files:
            mt = mimetypes.guess_type(p)[0] or "application/octet-stream"
            with open(p, "rb") as f:
                r = client.post(api + "/documents/upload", files={"file": (os.path.basename(p), f, mt)})
                r.raise_for_status()
                print("uploaded", os.path.basename(p))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

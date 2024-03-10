import re

def set_metadata(filename):
    keywords = []
    print(filename)
    with open(filename, "r", encoding="UTF-8") as filehandle:
        text = filehandle.read()
        for line in text.split("\n"):
            if "#" in line and len(line.split(" ")) < 10:
                keyword = line.replace("#", "")
                keyword = re.sub(r"^[ ]+", "", keyword)
                keywords.append(keyword)
    return {"keywords": keywords}
from bs4 import BeautifulSoup
import re


# Clean DOM
def clean_dom(raw_html : str) -> str :
    if not raw_html :
        return "Empty DOM"

    soup = BeautifulSoup(raw_html, "html.parser")

    # 1. Strip useless massive overhead tags
    for element in soup(["script", "style", "noscript", "svg", "path", "head", "meta", "iframe"]):
        element.extract()

    # 2. Clean out bulky inline attributes that bloat context (tailwinds CSS classes, styles)
    # Keeping 'id', 'class', 'href', 'name', 'type', and data-attributes for trading links
    allowed_attrs = ["id", "href", "name", "type", "value", "placeholder"]
    for tag in soup.find_all(True):
        # Create a copy of keys to mutate during iteration
        attrs = list(tag.attrs.keys())
        for attr in attrs:
            if attr not in allowed_attrs and not attr.startswith("data-"):
                del tag[attr]

    # 3. Collapse whitespace strings
    cleaned_text = soup.prettify()
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Collapse empty lines

    return cleaned_text


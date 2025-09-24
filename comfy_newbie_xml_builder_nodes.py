# src/nodes.py
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
try:
    from server import PromptServer
except Exception:
    PromptServer = None

def prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")

def safe_parse_fragment(fragment: str):
    fragment = (fragment or "").strip()
    if not fragment:
        return None
    try:
        return ET.fromstring(fragment)
    except Exception:
        return None

# -------------------------
# Character Node (multiline fields)
# -------------------------
class CharacterNode:
    CATEGORY = "Prompt/Builder"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING",),     # keep single-line for easy naming
                "gender": ("STRING",),
            },
            "optional": {
                # make these multiline textareas in the frontend
                "appearance": ("STRING", {"multiline": True}),
                "clothing": ("STRING", {"multiline": True}),
                "body_type": ("STRING", {"multiline": True}),
                "expression": ("STRING", {"multiline": True}),
                "action": ("STRING", {"multiline": True}),
                "interaction": ("STRING", {"multiline": True}),
                "position": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_character_xml"

    def build_character_xml(self, name, gender,
                            appearance="", clothing="", body_type="",
                            expression="", action="", interaction="", position=""):
        elem_name = (name or "character").strip()
        root = ET.Element(elem_name)
        ET.SubElement(root, "name").text = name or ""
        ET.SubElement(root, "gender").text = gender or ""
        ET.SubElement(root, "appearance").text = (appearance or "").strip()
        ET.SubElement(root, "clothing").text = (clothing or "").strip()
        ET.SubElement(root, "body_type").text = (body_type or "").strip()
        ET.SubElement(root, "expression").text = (expression or "").strip()
        ET.SubElement(root, "action").text = (action or "").strip()
        ET.SubElement(root, "interaction").text = (interaction or "").strip()
        ET.SubElement(root, "position").text = (position or "").strip()
        return (ET.tostring(root, encoding="unicode"),)

# -------------------------
# General Tag Builder (all multiline)
# -------------------------
class GeneralTagBuilder:
    CATEGORY = "Prompt/Builder"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "count": ("STRING", {"multiline": True}),
                "artists": ("STRING", {"multiline": True}),
                "style": ("STRING", {"multiline": True}),
                "background": ("STRING", {"multiline": True}),
                "environment": ("STRING", {"multiline": True}),
                "perspective": ("STRING", {"multiline": True}),
                "atmosphere": ("STRING", {"multiline": True}),
                "lighting": ("STRING", {"multiline": True}),
                "quality": ("STRING", {"multiline": True}),
                "objects": ("STRING", {"multiline": True}),
                "other": ("STRING", {"multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_general_tags"

    def build_general_tags(self, count=None, artists=None, style=None, background=None,
                           environment=None, perspective=None, atmosphere=None,
                           lighting=None, quality=None, objects=None, other=None):
        root = ET.Element("general_tags")
        def add_if(val, tag):
            child = ET.SubElement(root, tag)
            child.text = (val or "").strip()
        add_if(count, "count")
        add_if(artists, "artists")
        add_if(style, "style")
        add_if(background, "background")
        add_if(environment, "environment")
        add_if(perspective, "perspective")
        add_if(atmosphere, "atmosphere")
        add_if(lighting, "lighting")
        add_if(quality, "quality")
        add_if(objects, "objects")
        add_if(other, "other")
        return (ET.tostring(root, encoding="unicode"),)

# -------------------------
# XML Assembler (up to 10 chars)
# general_tags appended at BOTTOM
# -------------------------
class XMLAssembler10:
    CATEGORY = "Prompt/Builder"
    @classmethod
    def INPUT_TYPES(cls):
        # Only general_tags is required. All char fragments (1..10) are optional.
        required = {
            "general_tags": ("STRING",),  # still a STRING that will receive GeneralTagBuilder output
        }
        optional = {}
        # include char_fragment_1 .. char_fragment_10 as optional entries
        for i in range(1, 11):
            optional[f"char_fragment_{i}"] = ("STRING",)
        return {"required": required, "optional": optional}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "assemble_xml"

    def assemble_xml(self, general_tags,
                     char_fragment_1=None, char_fragment_2=None, char_fragment_3=None, char_fragment_4=None,
                     char_fragment_5=None, char_fragment_6=None, char_fragment_7=None,
                     char_fragment_8=None, char_fragment_9=None, char_fragment_10=None):
        fragments = [
            char_fragment_1, char_fragment_2, char_fragment_3, char_fragment_4,
            char_fragment_5, char_fragment_6, char_fragment_7, char_fragment_8,
            char_fragment_9, char_fragment_10
        ]
        root = ET.Element("document")
        for frag in fragments:
            if not frag:
                continue
            frag_text = (frag or "").strip()
            if not frag_text:
                continue
            parsed = safe_parse_fragment(frag_text)
            if parsed is not None:
                root.append(parsed)
            else:
                wrapper = ET.SubElement(root, "fragment")
                wrapper.text = frag_text

        # append general_tags at BOTTOM
        general_text = (general_tags or "").strip()
        if general_text:
            parsed_gt = safe_parse_fragment(general_text)
            if parsed_gt is not None and parsed_gt.tag == "general_tags":
                gt_elem = ET.SubElement(root, "general_tags")
                for child in parsed_gt:
                    new_child = ET.SubElement(gt_elem, child.tag)
                    new_child.text = (child.text or "").strip()
            else:
                gt_elem = ET.SubElement(root, "general_tags")
                gt_elem.text = general_text

        pretty = prettify_xml(root)

        # save fallback preview
        try:
            out_dir = os.path.join("outputs", "xml_prompts")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "latest_preview.xml"), "w", encoding="utf-8") as f:
                f.write(pretty)
            with open(os.path.join(out_dir, "latest_preview.html"), "w", encoding="utf-8") as f:
                f.write("<pre>" + pretty.replace('<','&lt;').replace('>','&gt;') + "</pre>")
        except Exception:
            pass

        # broadcast
        if PromptServer:
            try:
                PromptServer.instance.send_sync("xml_prompt_builder.preview", {"xml": pretty})
            except Exception:
                pass

        return (pretty,)

# Node registration map
NODE_CLASS_MAPPINGS = {
    "Character Node": CharacterNode,
    "General Tag Builder": GeneralTagBuilder,
    "XML Assembler (up to 10 chars)": XMLAssembler10,
}

"""
NewBie XML Prompt Builder
Comprehensive XML prompt builder for AI image generation with the NewBie format.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional
import os


def prettify_xml(elem: ET.Element, indent: str = "  ") -> str:
    """Return a pretty-printed XML string for the Element."""
    try:
        rough = ET.tostring(elem, encoding="utf-8")
        parsed = minidom.parseString(rough)
        pretty = parsed.toprettyxml(indent=indent)
        # Remove the XML declaration line
        lines = pretty.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        return '\n'.join(line for line in lines if line.strip())
    except Exception:
        return ET.tostring(elem, encoding="unicode")


def add_element_if_content(parent: ET.Element, tag: str, text: str) -> Optional[ET.Element]:
    """Add child element only if text has content."""
    text = (text or "").strip()
    if not text:
        return None
    child = ET.SubElement(parent, tag)
    child.text = text
    return child


# -------------------------
# Character Builder - Creates a single character XML block
# -------------------------
class NewBieCharacterBuilder:
    """
    Build a character XML block for NewBie prompts.
    Outputs XML like:
    <character_1>
      <n>character name</n>
      <gender>1girl</gender>
      ...
    </character_1>
    """
    
    CATEGORY = "NewBie/XMLBuilder"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "character_number": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Character number (1, 2, 3, etc.)"
                }),
                "name": ("STRING", {
                    "default": "",
                    "tooltip": "Character name (e.g., seia \\(blue archive\\))"
                }),
                "gender": ("STRING", {
                    "default": "1girl",
                    "tooltip": "Gender tag (1girl, 1boy, etc.)"
                }),
                "appearance": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Physical appearance description"
                }),
                "clothing": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Clothing and accessories"
                }),
                "expression": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Facial expression and mood"
                }),
                "action": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Pose, movement, gesture, activity"
                }),
                "interaction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Interaction with others/objects/environment"
                }),
                "position": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Position in frame (center, left, foreground, etc.)"
                }),
            }
        }
    
    RETURN_TYPES = ("CHARACTER_XML",)
    RETURN_NAMES = ("character",)
    FUNCTION = "build_character"
    
    def build_character(
        self,
        character_number: int = 1,
        name: str = "",
        gender: str = "1girl",
        appearance: str = "",
        clothing: str = "",
        expression: str = "",
        action: str = "",
        interaction: str = "",
        position: str = "",
    ):
        root = ET.Element(f"character_{character_number}")
        
        # <n> tag - always present, can be empty
        n_elem = ET.SubElement(root, "n")
        n_elem.text = name.strip() if name.strip() else ""
        
        # <gender> - always present
        gender_elem = ET.SubElement(root, "gender")
        gender_elem.text = gender.strip() if gender.strip() else "1girl"
        
        # Optional fields
        add_element_if_content(root, "appearance", appearance)
        add_element_if_content(root, "clothing", clothing)
        add_element_if_content(root, "expression", expression)
        add_element_if_content(root, "action", action)
        add_element_if_content(root, "interaction", interaction)
        add_element_if_content(root, "position", position)
        
        xml_str = ET.tostring(root, encoding="unicode")
        return (xml_str,)


# -------------------------
# General Tags Builder - Creates the general_tags XML block
# -------------------------
class NewBieGeneralTagsBuilder:
    """
    Build the general_tags XML block for NewBie prompts.
    """
    
    CATEGORY = "NewBie/XMLBuilder"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "count": ("STRING", {
                    "default": "1girl, solo",
                    "tooltip": "Total character count (1girl, 2girls, multiple boys, etc.)"
                }),
                "body_type": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Body type tags with weights"
                }),
                "artists": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Artist tags (e.g., artist:name, (artist1, artist2:1.5))"
                }),
                "style": ("STRING", {
                    "multiline": True,
                    "default": "anime style",
                    "tooltip": "Art style description"
                }),
                "background": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Background type (indoor, outdoor, etc.)"
                }),
                "environment": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Specific environment (room, forest, etc.)"
                }),
                "perspective": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Camera perspective (bird's-eye, low angle, etc.)"
                }),
                "atmosphere": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Mood and atmosphere"
                }),
                "lighting": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Lighting conditions"
                }),
                "quality": ("STRING", {
                    "multiline": True,
                    "default": "high quality illustration, clean lineart, no logo, no watermark",
                    "tooltip": "Quality tags"
                }),
                "objects": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Important objects in the scene"
                }),
                "extra_tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional danbooru tags"
                }),
            }
        }
    
    RETURN_TYPES = ("GENERAL_TAGS_XML",)
    RETURN_NAMES = ("general_tags",)
    FUNCTION = "build_general_tags"
    
    def build_general_tags(
        self,
        count: str = "1girl, solo",
        body_type: str = "",
        artists: str = "",
        style: str = "anime style",
        background: str = "",
        environment: str = "",
        perspective: str = "",
        atmosphere: str = "",
        lighting: str = "",
        quality: str = "high quality illustration, clean lineart, no logo, no watermark",
        objects: str = "",
        extra_tags: str = "",
    ):
        root = ET.Element("general_tags")
        
        add_element_if_content(root, "count", count)
        add_element_if_content(root, "body_type", body_type)
        add_element_if_content(root, "artists", artists)
        add_element_if_content(root, "style", style)
        add_element_if_content(root, "background", background)
        add_element_if_content(root, "environment", environment)
        add_element_if_content(root, "perspective", perspective)
        add_element_if_content(root, "atmosphere", atmosphere)
        add_element_if_content(root, "lighting", lighting)
        add_element_if_content(root, "quality", quality)
        add_element_if_content(root, "objects", objects)
        add_element_if_content(root, "extra_tags", extra_tags)
        
        xml_str = ET.tostring(root, encoding="unicode")
        return (xml_str,)


# -------------------------
# XML Prompt Assembler - Combines everything into final <image> XML
# -------------------------
class NewBieXMLPromptAssembler:
    """
    Assemble the complete XML prompt with caption, characters, and general_tags.
    Outputs the final <image>...</image> XML block.
    """
    
    CATEGORY = "NewBie/XMLBuilder"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Natural language description of the scene"
                }),
            },
            "optional": {
                "character_1": ("CHARACTER_XML", {"tooltip": "First character XML"}),
                "character_2": ("CHARACTER_XML", {"tooltip": "Second character XML"}),
                "character_3": ("CHARACTER_XML", {"tooltip": "Third character XML"}),
                "character_4": ("CHARACTER_XML", {"tooltip": "Fourth character XML"}),
                "general_tags": ("GENERAL_TAGS_XML", {"tooltip": "General tags XML"}),
                "style_prefix": ("STRING", {
                    "default": "style of mikozin",
                    "tooltip": "Style prefix (wrapped in <style> tag at the top)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("xml_prompt",)
    OUTPUT_TOOLTIPS = ("Complete XML prompt ready for use",)
    FUNCTION = "assemble"
    
    def assemble(
        self,
        caption: str,
        character_1: str = None,
        character_2: str = None,
        character_3: str = None,
        character_4: str = None,
        general_tags: str = None,
        style_prefix: str = "style of mikozin",
    ):
        root = ET.Element("image")
        
        # Add caption first
        caption_elem = ET.SubElement(root, "caption")
        caption_elem.text = caption.strip() if caption else ""
        
        # Add characters in order
        for char_xml in [character_1, character_2, character_3, character_4]:
            if char_xml:
                try:
                    char_elem = ET.fromstring(char_xml)
                    root.append(char_elem)
                except ET.ParseError:
                    pass
        
        # Add general_tags
        if general_tags:
            try:
                gt_elem = ET.fromstring(general_tags)
                root.append(gt_elem)
            except ET.ParseError:
                pass
        
        # Build final output with style prefix
        xml_content = prettify_xml(root)
        
        # Add style prefix at the top if provided
        if style_prefix and style_prefix.strip():
            style_line = f"<style>{style_prefix.strip()}</style>\n\n"
            final_output = style_line + xml_content
        else:
            final_output = xml_content
        
        # Save preview
        try:
            out_dir = os.path.join("outputs", "xml_prompts")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "latest_newbie_prompt.xml"), "w", encoding="utf-8") as f:
                f.write(final_output)
        except Exception:
            pass
        
        return (final_output,)


# -------------------------
# Quick Character - Simplified single-input character builder
# -------------------------
class NewBieQuickCharacter:
    """
    Quick character builder - paste all details in one text block.
    Parses sections marked with [name], [gender], [appearance], etc.
    """
    
    CATEGORY = "NewBie/XMLBuilder"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_number": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                }),
                "character_text": ("STRING", {
                    "multiline": True,
                    "default": """[name] character name
[gender] 1girl
[appearance] description here
[clothing] outfit description
[expression] expression and mood
[action] pose and activity
[interaction] interactions
[position] position in frame""",
                    "tooltip": "Paste character details with [section] markers"
                }),
            }
        }
    
    RETURN_TYPES = ("CHARACTER_XML",)
    RETURN_NAMES = ("character",)
    FUNCTION = "parse_character"
    
    def parse_character(self, character_number: int, character_text: str):
        import re
        
        # Parse sections
        sections = {
            "name": "",
            "gender": "1girl",
            "appearance": "",
            "clothing": "",
            "expression": "",
            "action": "",
            "interaction": "",
            "position": "",
        }
        
        # Find all [section] content patterns
        pattern = r'\[(\w+)\]\s*(.*?)(?=\[\w+\]|$)'
        matches = re.findall(pattern, character_text, re.DOTALL | re.IGNORECASE)
        
        for section_name, content in matches:
            section_name = section_name.lower().strip()
            content = content.strip()
            if section_name in sections:
                sections[section_name] = content
        
        # Build XML
        root = ET.Element(f"character_{character_number}")
        
        n_elem = ET.SubElement(root, "n")
        n_elem.text = sections["name"]
        
        gender_elem = ET.SubElement(root, "gender")
        gender_elem.text = sections["gender"]
        
        for field in ["appearance", "clothing", "expression", "action", "interaction", "position"]:
            if sections[field]:
                add_element_if_content(root, field, sections[field])
        
        xml_str = ET.tostring(root, encoding="unicode")
        return (xml_str,)


# -------------------------
# Text to XML Prompt - All-in-one simple converter
# -------------------------
class NewBieTextToXMLPrompt:
    """
    Simple all-in-one node: Takes natural text inputs and builds complete XML prompt.
    Good for quick single-character prompts.
    """
    
    CATEGORY = "NewBie/XMLBuilder"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Natural language scene description"
                }),
            },
            "optional": {
                # Character 1
                "char1_name": ("STRING", {"default": ""}),
                "char1_gender": ("STRING", {"default": "1girl"}),
                "char1_appearance": ("STRING", {"multiline": True, "default": ""}),
                "char1_clothing": ("STRING", {"multiline": True, "default": ""}),
                "char1_expression": ("STRING", {"multiline": True, "default": ""}),
                "char1_action": ("STRING", {"multiline": True, "default": ""}),
                
                # General
                "count": ("STRING", {"default": "1girl, solo"}),
                "body_type": ("STRING", {"multiline": True, "default": ""}),
                "artists": ("STRING", {"multiline": True, "default": ""}),
                "style": ("STRING", {"multiline": True, "default": "anime style"}),
                "background": ("STRING", {"multiline": True, "default": ""}),
                "lighting": ("STRING", {"multiline": True, "default": ""}),
                "quality": ("STRING", {"default": "high quality illustration, no watermark"}),
                "extra_tags": ("STRING", {"multiline": True, "default": ""}),
                
                "style_prefix": ("STRING", {
                    "default": "style of mikozin"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("xml_prompt",)
    FUNCTION = "build_prompt"
    
    def build_prompt(
        self,
        caption: str,
        char1_name: str = "",
        char1_gender: str = "1girl",
        char1_appearance: str = "",
        char1_clothing: str = "",
        char1_expression: str = "",
        char1_action: str = "",
        count: str = "1girl, solo",
        body_type: str = "",
        artists: str = "",
        style: str = "anime style",
        background: str = "",
        lighting: str = "",
        quality: str = "high quality illustration, no watermark",
        extra_tags: str = "",
        style_prefix: str = "style of mikozin",
    ):
        root = ET.Element("image")
        
        # Caption
        caption_elem = ET.SubElement(root, "caption")
        caption_elem.text = caption.strip() if caption else ""
        
        # Character 1
        char1 = ET.SubElement(root, "character_1")
        n_elem = ET.SubElement(char1, "n")
        n_elem.text = char1_name.strip()
        gender_elem = ET.SubElement(char1, "gender")
        gender_elem.text = char1_gender.strip() or "1girl"
        add_element_if_content(char1, "appearance", char1_appearance)
        add_element_if_content(char1, "clothing", char1_clothing)
        add_element_if_content(char1, "expression", char1_expression)
        add_element_if_content(char1, "action", char1_action)
        
        # General tags
        gt = ET.SubElement(root, "general_tags")
        add_element_if_content(gt, "count", count)
        add_element_if_content(gt, "body_type", body_type)
        add_element_if_content(gt, "artists", artists)
        add_element_if_content(gt, "style", style)
        add_element_if_content(gt, "background", background)
        add_element_if_content(gt, "lighting", lighting)
        add_element_if_content(gt, "quality", quality)
        add_element_if_content(gt, "extra_tags", extra_tags)
        
        xml_content = prettify_xml(root)
        
        if style_prefix and style_prefix.strip():
            final_output = f"<style>{style_prefix.strip()}</style>\n\n{xml_content}"
        else:
            final_output = xml_content
        
        return (final_output,)


# -------------------------
# Character Combiner - Combines multiple character XMLs
# -------------------------
class NewBieCharacterCombiner:
    """
    Combine up to 6 character XMLs into a list for the assembler.
    Also renumbers them sequentially.
    """
    
    CATEGORY = "NewBie/XMLBuilder"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "char_1": ("CHARACTER_XML",),
                "char_2": ("CHARACTER_XML",),
                "char_3": ("CHARACTER_XML",),
                "char_4": ("CHARACTER_XML",),
                "char_5": ("CHARACTER_XML",),
                "char_6": ("CHARACTER_XML",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_characters",)
    FUNCTION = "combine"
    
    def combine(
        self,
        char_1: str = None,
        char_2: str = None,
        char_3: str = None,
        char_4: str = None,
        char_5: str = None,
        char_6: str = None,
    ):
        characters = []
        char_num = 1
        
        for char_xml in [char_1, char_2, char_3, char_4, char_5, char_6]:
            if char_xml:
                try:
                    # Parse and renumber
                    elem = ET.fromstring(char_xml)
                    elem.tag = f"character_{char_num}"
                    characters.append(ET.tostring(elem, encoding="unicode"))
                    char_num += 1
                except ET.ParseError:
                    pass
        
        combined = "\n".join(characters)
        return (combined,)


NODE_CLASS_MAPPINGS = {
    "NewBieCharacterBuilder": NewBieCharacterBuilder,
    "NewBieGeneralTagsBuilder": NewBieGeneralTagsBuilder,
    "NewBieXMLPromptAssembler": NewBieXMLPromptAssembler,
    "NewBieQuickCharacter": NewBieQuickCharacter,
    "NewBieTextToXMLPrompt": NewBieTextToXMLPrompt,
    "NewBieCharacterCombiner": NewBieCharacterCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieCharacterBuilder": "NewBie Character Builder",
    "NewBieGeneralTagsBuilder": "NewBie General Tags Builder",
    "NewBieXMLPromptAssembler": "NewBie XML Prompt Assembler",
    "NewBieQuickCharacter": "NewBie Quick Character",
    "NewBieTextToXMLPrompt": "NewBie Text to XML Prompt",
    "NewBieCharacterCombiner": "NewBie Character Combiner",
}

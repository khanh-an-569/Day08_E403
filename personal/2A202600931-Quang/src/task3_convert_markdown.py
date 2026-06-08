import os
import json
from markitdown import MarkItDown

def convert_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    md = MarkItDown()
    
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        
        if os.path.isdir(input_path):
            continue
            
        if filename.startswith('.'):
            continue
            
        md_text = ""
        # Đối với file json do Task 2 sinh ra (đã có sẵn markdown trong field 'content')
        if filename.endswith('.json'):
            print(f"Extracting markdown from {input_path}...")
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    md_text = f"# {data.get('url', '')}\n\n{data.get('content', '')}"
            except Exception as e:
                print(f"Failed to read JSON {input_path}: {e}")
                continue
        else:
            # Đối với các file khác (pdf, docx, txt...) dùng MarkItDown
            print(f"Converting {input_path} using MarkItDown...")
            try:
                result = md.convert(input_path)
                md_text = result.text_content
            except Exception as e:
                print(f"Failed to convert {input_path}: {e}")
                continue
                
        # Generate output filename (change extension to .md)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.md"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_text)
            print(f"  -> Saved to {output_path}")
        except Exception as e:
            print(f"  -> Failed to save {output_path}: {e}")

if __name__ == "__main__":
    landing_legal = os.path.join("data", "landing", "legal")
    standardized_legal = os.path.join("data", "standardized", "legal")
    print("--- Converting Legal Docs ---")
    convert_directory(landing_legal, standardized_legal)
    
    landing_news = os.path.join("data", "landing", "news")
    standardized_news = os.path.join("data", "standardized", "news")
    print("\n--- Converting News Docs ---")
    convert_directory(landing_news, standardized_news)

import markdown
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import argparse
import re

def convert_md_to_pdf(input_file: str, output_file: str = None):
    """
    Convert a Markdown file to PDF using ReportLab.
    
    Args:
        input_file (str): Path to the input Markdown file
        output_file (str): Path for the output PDF file
    """
    # Read markdown content
    with open(input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'smarty', 'tables', 'codehilite']
    )
    
    # If no output file specified, use input filename with .pdf extension
    if output_file is None:
        output_file = input_file.rsplit('.', 1)[0] + '.pdf'
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create or update custom styles
    code_style = ParagraphStyle(
        name='CustomCode',  # Changed from 'Code' to avoid conflict
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        leading=12,
        leftIndent=36
    )
    styles.add(code_style)
    
    # Process HTML content into PDF elements
    story = []
    
    # Split content into blocks
    blocks = re.split(r'(<h[1-6]>.*?</h[1-6]>|<pre><code>.*?</code></pre>|<p>.*?</p>)', 
                     html_content, flags=re.DOTALL)
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Handle headers
        if block.startswith('<h'):
            level = int(block[2])
            text = re.sub('<[^<]+?>', '', block)
            style = styles[f'Heading{level}']
            story.append(Paragraph(text, style))
            story.append(Spacer(1, 12))
            
        # Handle code blocks
        elif block.startswith('<pre><code>'):
            text = re.sub('<[^<]+?>', '', block)
            story.append(Preformatted(text, styles['CustomCode']))
            story.append(Spacer(1, 12))
            
        # Handle paragraphs
        elif block.startswith('<p>'):
            text = re.sub('<[^<]+?>', '', block)
            story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting to PDF: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown file to PDF")
    parser.add_argument("input_file", help="Path to input Markdown file")
    parser.add_argument("-o", "--output", help="Path to output PDF file")
    
    args = parser.parse_args()
    convert_md_to_pdf(args.input_file, args.output)
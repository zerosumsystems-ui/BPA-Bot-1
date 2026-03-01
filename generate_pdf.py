from fpdf import FPDF
import sys

class PDF(FPDF):
    def header(self):
        # Setting a corporate aesthetic
        self.set_fill_color(10, 40, 80) # Dark Blue Header
        self.rect(0, 0, 210, 20, 'F')
        
        self.set_y(8)
        self.set_font('Arial', 'B', 15)
        self.set_text_color(255, 255, 255)
        self.cell(0, 5, 'QUANTITATIVE RESEARCH', 0, 1, 'L')
        
        self.set_y(8)
        self.set_font('Arial', 'I', 11)
        self.cell(0, 5, 'Internal Distribution Only', 0, 1, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf():
    pdf = PDF()
    pdf.add_page()
    
    with open('/Users/williamkosloski/BPA-Bot-1/Algorithm_Explanations.md', 'r') as f:
        text = f.read()

    lines = text.split('\n')
    
    # Title parsing
    for line in lines:
        if line.startswith('# '):
            pdf.set_font('Arial', 'B', 16)
            pdf.set_text_color(10, 40, 80) # Dark Blue
            pdf.multi_cell(0, 10, line.replace('# ', '').strip())
            pdf.ln(5)
        elif line.startswith('## '):
            pdf.set_font('Arial', 'B', 14)
            pdf.set_text_color(20, 20, 20)
            pdf.ln(5)
            pdf.multi_cell(0, 8, line.replace('## ', '').strip())
            # Add subtle underline to headers
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.get_x(), pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
        elif line.startswith('**'):
            pdf.set_font('Arial', 'B', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 8, line.replace('**', ''))
        elif line == '---':
            pdf.ln(5)
            pdf.set_draw_color(10, 40, 80)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
        elif line.strip() != "":
            pdf.set_font('Arial', '', 11)
            pdf.set_text_color(50, 50, 50)
            
            # Very basic inline bold parsing for "Module:" and "Target Environment:"
            if "**" in line:
                parts = line.split("**")
                if len(parts) >= 3:
                     pdf.set_font('Arial', 'B', 11)
                     pdf.write(6, parts[1])
                     pdf.set_font('Arial', '', 11)
                     pdf.write(6, parts[2] + "\n")
                     continue
            
            pdf.multi_cell(0, 6, line)
            pdf.ln(2)

    pdf.output('/Users/williamkosloski/BPA-Bot-1/Algorithm_Explanations.pdf')
    print("Goldman Sachs Style PDF Successfully Created at /Users/williamkosloski/BPA-Bot-1/Algorithm_Explanations.pdf")

if __name__ == "__main__":
    try:
        create_pdf()
    except Exception as e:
        print(f"Error creating PDF: {e}")
        sys.exit(1)

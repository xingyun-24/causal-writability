#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
from PIL import Image

EXPECTED={'freefall_behavior':(3862,2419),'freefall_state_geometry':(3844,1831),'freefall_decoded_recovery':(3887,1983),'freefall_writeability':(3848,1768)}
PDFTOPPM=Path(r'C:\Users\34655\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
PDFFONTS=Path(r'C:\texlive\2025\bin\windows\pdffonts.exe')

def main():
    p=argparse.ArgumentParser();p.add_argument('kit',type=Path);p.add_argument('--render-dir',type=Path,required=True);a=p.parse_args();a.render_dir.mkdir(parents=True,exist_ok=True)
    report={'figures':{},'all_pass':True}
    for stem,size in EXPECTED.items():
        png=a.kit/'figures'/f'{stem}.png';pdf=a.kit/'figures'/f'{stem}.pdf';svg=a.kit/'figures'/f'{stem}.svg'
        image=Image.open(png);dpi=image.info.get('dpi',(0,0));svg_text=svg.read_text(encoding='utf-8')
        prefix=a.render_dir/stem;subprocess.run([str(PDFTOPPM),'-f','1','-singlefile','-r','150','-png',str(pdf),str(prefix)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        font_text=subprocess.run([str(PDFFONTS),str(pdf)],check=True,text=True,capture_output=True).stdout
        font_lines=[line for line in font_text.splitlines()[2:] if line.strip()]
        embedded=all((' yes ' in f' {line} ') for line in font_lines)
        item={'png_pixels':list(image.size),'expected_pixels':list(size),'png_dpi':list(dpi),'pixel_size_pass':image.size==size,
              'svg_contains_editable_text':'<text' in svg_text,'pdf_fonts':font_lines,'pdf_fonts_embedded':embedded,'rendered_pdf_png':str((prefix.with_suffix('.png')).resolve())}
        item['pass']=item['pixel_size_pass'] and item['svg_contains_editable_text'] and embedded
        report['figures'][stem]=item;report['all_pass']&=item['pass']
    (a.kit/'figure_QA.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));assert report['all_pass']

if __name__=='__main__':main()

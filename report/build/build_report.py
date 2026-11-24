import re, markdown, base64, pathlib
B = pathlib.Path("/tmp/claude-0/brand")
md = (B / "report.md").read_text()
body = markdown.markdown(md, extensions=["tables", "md_in_html", "attr_list"])
# split "N Title" headings into a yellow number chip + title
body = re.sub(r"<h2>(\d+) ([^<]+)</h2>", r'<h2><span class="num">\1</span>\2</h2>', body)
body = re.sub(r"<h3>(\d+\.\d+) ([^<]+)</h3>", r'<h3><span class="n3">\1</span>\2</h3>', body)
toc = "".join(f'<li><span class="tn">{n}</span>{t}</li>' for n, t in re.findall(r'<h2><span class="num">(\d+)</span>([^<]+)</h2>', body))
crest = base64.b64encode((B / "crest.png").read_bytes()).decode()
CSS = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm; }
* { box-sizing: border-box; }
html { font-family: 'Inter', Arial, sans-serif; font-size: 10pt; color: #231F20; }
body { margin: 0; line-height: 1.55; }
p { margin: 0 0 3.2mm; text-align: left; }
h2 { font-size: 17pt; font-weight: 700; margin: 9mm 0 4mm; display: flex; align-items: center; gap: 3mm;
     break-after: avoid; letter-spacing: -0.2px; }
h2 .num { font-family: 'Roboto Mono', monospace; font-size: 11pt; background: #FFDC00; color: #231F20;
          padding: 1.2mm 2.4mm; border-radius: 1.2mm; font-weight: 700; }
h2.newpage { break-before: page; margin-top: 0; }
h3 { font-size: 12pt; font-weight: 700; margin: 6mm 0 2.5mm; break-after: avoid; }
h3 .n3 { font-family: 'Roboto Mono', monospace; font-size: 10pt; color: #6b6b6b; margin-right: 2.5mm; font-weight: 500; }
strong, b { font-weight: 700; }
ul, ol { margin: 0 0 3.5mm; padding-left: 6mm; } li { margin-bottom: 1.2mm; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 5mm; font-size: 8.8pt; line-height: 1.4; break-inside: avoid; }
th { background: #231F20; color: #fff; text-align: left; font-weight: 600; padding: 2mm 2.5mm; }
td { padding: 1.8mm 2.5mm; border-bottom: 0.3mm solid #e2e2e2; vertical-align: top; }
tr:nth-child(even) td { background: #f7f7f7; }
figure { margin: 4mm 0 6mm; break-inside: avoid; }
figure img { width: 100%; display: block; }
figure:not(.wide) img { width: 88%; margin: 0 auto; }
figcaption { font-size: 8.6pt; color: #555; margin-top: 2mm; line-height: 1.45; }
.eq { font-size: 13pt; text-align: center; margin: 3mm 0 5mm; font-weight: 600; }
.abstract { background: #FFF6C2; padding: 6mm 7mm 3mm; border-radius: 2mm; }
.abstract p { margin-bottom: 3mm; }
.keyfig { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; margin: 5mm 0 2mm; }
.keyfig div { background: #231F20; color: #fff; border-radius: 2mm; padding: 4mm 3.5mm; }
.keyfig b { display: block; font-size: 15pt; color: #FFDC00; letter-spacing: -0.3px; }
.keyfig span { display: block; font-size: 8pt; color: #d4d4d4; margin-top: 1.5mm; line-height: 1.35; }
.refs p { font-size: 8.6pt; padding-left: 6mm; text-indent: -6mm; margin-bottom: 2.2mm; line-height: 1.45; }
.toc { list-style: none; padding: 0; margin: 0 0 6mm; columns: 2; column-gap: 10mm; }
.toc li { margin: 0 0 2mm; font-size: 10pt; }
.toc .tn { font-family: 'Roboto Mono', monospace; display: inline-block; width: 8mm; color: #6b6b6b; }
.toc-h { font-family: 'Roboto Mono', monospace; font-size: 9pt; letter-spacing: 1px; color: #6b6b6b; margin: 0 0 3mm; }
"""
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="toc-h">CONTENTS</div><ul class="toc">{toc}<li><span class="tn">—</span>References</li></ul>
{body}</body></html>"""
html = html.replace('<h2 class="abstract-h">Abstract</h2>', '<h2>Abstract</h2>')
html = html.replace('<h2 class="refs-h">References</h2>', '<h2 class="newpage">References</h2>')
html = html.replace('<h2><span class="num">1</span>', '<h2 class="newpage"><span class="num">1</span>')
(B / "report_body.html").write_text(html)

COVER = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 0; }} * {{ box-sizing: border-box; }}
html {{ font-family: 'Inter', Arial, sans-serif; color: #231F20; }} body {{ margin: 0; }}
.page {{ width: 210mm; height: 297mm; position: relative; overflow: hidden; }}
.flag {{ background: #FFDC00; height: 118mm; padding: 16mm 20mm 0; position: relative; }}
.flag img {{ width: 30mm; }}
.kicker {{ font-family: 'Roboto Mono', monospace; font-size: 10pt; letter-spacing: 1.5px; position: absolute; left: 20mm; bottom: 14mm; }}
.title {{ padding: 16mm 20mm 0; }}
h1 {{ font-size: 34pt; line-height: 1.08; margin: 0 0 6mm; letter-spacing: -0.8px; font-weight: 800; }}
.sub {{ font-size: 14pt; color: #444; line-height: 1.4; margin: 0; max-width: 150mm; }}
.meta {{ position: absolute; left: 20mm; right: 20mm; bottom: 22mm; display: grid; grid-template-columns: 1fr 1fr; gap: 6mm 12mm; }}
.meta div span {{ display: block; font-family: 'Roboto Mono', monospace; font-size: 8pt; letter-spacing: 1px; color: #777; margin-bottom: 1mm; }}
.meta div b {{ font-size: 11pt; font-weight: 600; }}
.rule {{ position: absolute; left: 20mm; right: 20mm; bottom: 62mm; border-top: 0.4mm solid #231F20; }}
</style></head><body><div class="page">
<div class="flag"><img src="data:image/png;base64,{crest}"><div class="kicker">TASTE OF RESEARCH · TECHNICAL REPORT · 2026</div></div>
<div class="title"><h1>Bubble Size Measurement in a Novel Flotation Technology</h1>
<p class="sub">Deep-learning measurement of bubble size and Sauter mean diameter (D32) from pulp and froth video, packaged as the BubbleMetrix desktop tool</p></div>
<div class="rule"></div>
<div class="meta">
<div><span>AUTHOR</span><b>Tristan (Quoc Anh) Trieu · z5793457</b></div>
<div><span>SUPERVISOR</span><b>Prof Seher Ata</b></div>
<div><span>SCHOOL</span><b>Minerals &amp; Energy Resources Engineering, UNSW Sydney</b></div>
<div><span>PROGRAM · DATE</span><b>Faculty of Engineering Taste of Research · November 2026</b></div>
</div></div></body></html>"""
(B / "report_cover.html").write_text(COVER)
print("html ok")

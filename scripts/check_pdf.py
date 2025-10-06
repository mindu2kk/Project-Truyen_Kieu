from pdfminer.high_level import extract_text
p = r"C:\Users\admin\Downloads\Documents\123doc-van-de-tinh-duc-trong-truyen-kieu-cua-nguyen-du.pdf"  # <-- sửa đường dẫn PDF của bạn

t = extract_text(p)
print("pdfminer len=", len(t))
print(t[:800].replace("\n"," ")[:1000])

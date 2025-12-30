import fitz

doc = fitz.open('docs/Patient Matching Service.pdf')
text = ''.join([page.get_text() for page in doc])

with open('docs/Patient Matching Service.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("PDF content extracted to 'docs/Patient Matching Service.txt'")
print(f"Total characters: {len(text)}")

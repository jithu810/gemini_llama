SOF_QUESTION= """
You are a document data extraction assistant. Extract only the main tabular event data from the provided image exactly as it appears, without any interpretation, completion, or transformation.
The main table contains:
- An Event Name (may include terms like "Port log")
- A Date (either a single date or a range like "04/06/2023 to 05/06/2023")
- A Time (either a single time or a range like "10:00 to 18:00")
 Important:
- Do **not** fill in missing values.
- Do **not** guess or complete missing dates or times.
- If a field is **blank or missing** in the table, leave it **out of the output entirely**.
 Absolutely avoid adding any default values, assumptions, or inferred content.
 Only extract data from the **main event log table**. Ignore all other tables or unrelated tabular data.
 Return the data in this format **only when the values are present**:
[
  { "portlog": "End of Seapassage", "From date": "04/06/2023", "time": "15:00", "To date": "05/06/2023", "time": "15:00" },
  { "portlog": "N.O.R. tendered", "From date": "04/06/2023", "time": "15:35" },
  { "portlog": "Port log change", "From date": "05/06/2023 to 06/06/2023", "time": "10:00 to 18:00" }
]
 Notes:
- Only include the fields that are explicitly visible in the table.
- Maintain the row order as seen.
- Ignore all headers, footers, remarks, stamps, or unrelated sections.
"""

SUMMARIZE_PROMPT="""
You are a document assistant. Use the following content to answer user queries
"""

DEFAULT_SUMMARIZE_PROMPT="""
summarize the given content and give meaningful insights
"""

REFINE_TEXT_PROMPT="""
you act like a english grammer expert convert the text to a refined form withgout changing any context or meaning the user gave and dont add any Note
at the end just give the email itself
"""

COMMON_INVOICE_PROMPT = """Please extract only the specific information requested. Provide just the exact 
details without any extra context or explanation. what is the?"""

MATCHING_PROMPT="""
Analyze the provided image and extract the total amount mentioned in it.Ensure that you return only the numerical value, 
including commas and decimal points if present Ignore any irrelevant text. If multiple amounts are present, 
extract the final or total amount.Return only the extracted value without additional text.
If the total amount is in EUR format convert it to standard and give final
"""

BASE_RAG_SYSTEM_PROMPT = """
You are a knowledgeable and helpful assistant. Use the following retrieved context snippets to answer the user's question as accurately and concisely as possible.
If the answer is not contained in the context, respond politely that you don't know.
Context:
{context}
Please provide a clear and informative response.
""" 

DIFFICULTY_TABLE_1="""
Image contains unordered table which is split only extract the first table from the image".
For each row, extract the following fields:
- "id": Voucher number (far-left column, e.g., 1, 2, 3)
- "item_name": Item title (bold header above description, e.g., "AGENCY FEE")
- "description": Subtext under item name (e.g., "BANK CHARGES")
- "invoice_number": Appears in the invoice column (e.g., VOUCHER #20202996)
- "amount_in_PLN": Polish currency (e.g., 125,26 zł) — use only the number
- "amount_in_EUR": Euro value (e.g., € 325,49) — use only the number
Return the result as a JSON list of objects like:
[
  {
    "id": "",
    "item_name": "",
    "description": "",
    "invoice_number": "",
    "amount_in_PLN": "",
    "amount_in_EUR": ""
  },
  ...
]
Only include entries from the **first table** before the section titled "CREW MATTERS".
"""

DIFFICULTY_TABLE_2="""
From the image, extract only the second section titled "CREW MATTERS" starting from row 4 to row 17.
For each row, extract the following fields:
- "id": Voucher number (far-left column, e.g., 4, 5, 6)
- "item_name": Item title (bold header, e.g., "TAXI EXPENSES")
- "description": Text below item name (e.g., "AIRPORT - HOTEL TRANSFERS...")
- "invoice_number": From the invoice number column (e.g., 9/09/2022)
- "amount_in_PLN": Polish amount (e.g., 285,00 zł) — extract numeric part only
- "amount_in_EUR": Euro amount (e.g., € 62,67) — extract numeric part only
Return as a JSON list of objects:
[
  {
    "id": "",
    "item_name": "",
    "description": "",
    "invoice_number": "",
    "amount_in_PLN": "",
    "amount_in_EUR": ""
  },
  ...
]
Only include rows **after the "CREW MATTERS" heading**.
"""

OUTKERK_PROMPT="""
From the invoice image, extract all service line items where a description is followed by an amount. Ignore headings, totals, payment terms, or other non-line-item text.
Return the output in the following JSON format:
[
  {"description": "string", "amount": float},
  ...
]
Rules:
Only extract line items that have a textual description and a numerical amount.
The amount may be in European format (e.g., 2.557,00), where:
. is the thousands separator
, is the decimal separator
Convert such amounts into standard float format (e.g., 2.557,00 → 2557.00, 35,00 → 35.00)
Do not include totals or section headers in the output.
"""

EXCEL_COLUMN_PROMPT="""
You are given two lists of column headers from two datasets. Your task is to compare them and identify semantically similar or matching columns, even if the wording is different (e.g., "sex" and "gender" should be treated as similar).

Instructions:
- Return ONLY a JSON object.
- Do NOT include any Python code, explanations, or formatting tags like ```python.
- Do NOT generate a function or script.
- Your response must start and end with a JSON object, nothing else.

The JSON structure should be:
{
  "matched": [
    {"list1": "<header_from_list1>", "list2": "<similar_header_from_list2>"},
    ...
  ],
  "unmatched_list1": ["<header_from_list1>", ...],
  "unmatched_list2": ["<header_from_list2>", ...]
}

Use exact strings from the input lists, and match based on synonym meaning or common usage (e.g., "dob" ≈ "date of birth").
"""

WEBSITE_PROMPT="""
You are given a block of text containing information about a ship. Your task is to extract structured details for two sections: **Voyage Data** and **Recent Port Calls**.
Instructions:
- Extract only the required fields and return the result strictly as a **JSON object**.
- Do not include any explanation, description, or code.
- Do not wrap the response in a Python code block.
- Just return a raw JSON object with this structure:
{
  "voyage_data": {
    "Destination": "...",
    "ETA": "...",
    "Predicted ETA": "...",
    "Distance / Time": "...",
    "Course / Speed": "...",
    "Current draught": "...",
    "Navigation Status": "...",
    "Position received": "...",
    "Last Port": "...",
    "ATD": "..."
  },
  "recent_port_calls": [
    {
      "Port Name": "...",
      "Arrival (UTC)": "...",
      "Departure (UTC)": "...",
      "Port Stay": "..."
    }
  ]
}
- If a value is missing, not available, or marked as N/A, use `null`.
- Again, return only the JSON. Do not return Python code or any text around it.
"""

WEBSITE_PROMPT2 = """
You are given a block of text containing information about a vessel's journey. Your task is to extract the structured details into a JSON object with two main sections: **Voyage Summary** and **Trip Metrics**.

Instructions:
- Extract only the required fields and return the result strictly as a **JSON object**.
- Do not include any explanation, description, or code.
- Do not wrap the response in a Python code block.
- Just return a raw JSON object with this structure:

{
  "voyage_summary": {
    "Departure Port": "...",
    "Departure Country Flag": "...",
    "ATD": "...",
    "Arrival Port": "...",
    "Arrival Country Flag": "...",
    "ATA": "...",
    "Arrival Status": "..."  // e.g., "2 days ago"
  },
  "trip_metrics": {
    "Trip Time": "...",
    "Trip Distance (nm)": "...",
    "Average Speed (knots)": "...",
    "Max Speed (knots)": "...",
    "Draught (m)": "...",
    "Average Wind (knots)": "...",
    "Max Wind (knots)": "...",
    "Min Temperature (°C / °F)": "...",
    "Max Temperature (°C / °F)": "...",
    "Position Received": "..."
  }
}
- Use the exact keys shown above.
- If a value is missing, not available, or marked as N/A, use `null`.
- Return only the JSON object, no extra formatting.
"""

BOF_PROMPT="""You are given a Bill of Lading document. Extract key fields and return a single JSON object in the following format. Do not include tables or goods descriptions. Return nothing but the JSON.
Format strictly like this:
{
  "shipper": "Shipper name, address, contact",
  "consignee": "Consignee name, address, email, port, country, phone",
  "notify_party": "Notify Party name, address, email, port, country, phone",
  "carrier": "",
  "place_of_receipt": "",
  "port_of_loading": "",
  "vessel": "",
  "port_of_discharge": "",
  "place_of_delivery": "",
  "bill_of_lading_no": "",
  "voyage_no": "",
  "number_of_bills_of_lading": "",
  "jurisdiction_clause_present": true
}
nstructions:
Combine multiline fields into a single line per section (e.g., shipper).
If a value is not found, use an empty string "".
The jurisdiction_clause_present must be true if any legal/jurisdiction clause is found, else false.
Do not extract the item details or measurements.
"""

BOF_TABLE="""
This is a shipping document (Bill of Lading). Please extract the entire table from the image, including:

- Container Number
- Seal Number
- Marks & Numbers
- Number of Packages
- Description of Goods
- Gross Weight
- Net Weight
- Measurement (CBM)
- Agent Name, Address, Phone, Email

Output the data as a structured JSON object, with appropriate field names.
"""

BOF_THIRD_PROMPT = """
You are given a Bill of Lading document. Extract ONLY the number of original bills mentioned in the document.
"""

BOF_PROMPT_TEST="""You are given a Bill of Lading document. Extract all the Container No from the image"""

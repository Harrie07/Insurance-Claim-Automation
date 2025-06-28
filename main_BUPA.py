import os, re
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import time

# Flask App
app = Flask(__name__)

# Free API tokens (get from Hugging Face)
HF_TOKEN = "your api key"

# Explicitly specify model for local fallback
try:
    from transformers import pipeline
    # Use a smaller, efficient model
    local_qa = pipeline('question-answering', model="distilbert-base-cased-distilled-squad")
except ImportError:
    local_qa = None
except Exception as e:
    print(f"Error loading local model: {str(e)}")
    local_qa = None

general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease", 
                         "pregnancy", "substance abuse", "self-inflicted injuries", 
                         "sexually transmitted diseases(std)", "pre-existing conditions"]

# Policy document cache
policy_docs = {
    "claim_approval": "Documents required: ID proof, medical bills, doctor's prescription, hospital discharge summary.",
    "exclusions": "General exclusions: " + ", ".join(general_exclusion_list)
}

def get_file_content(file):
    """Enhanced PDF text extraction with better error handling"""
    text = ""
    if file.filename.endswith(".pdf"):
        try:
            pdf = PdfReader(file)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return ""
    return text.strip()

def extract_expense_with_regex(text):
    """Enhanced regex method for expense extraction with better patterns"""
    import re
    
    print(f"Extracting from text: {text}")
    
    # Split text into lines for better processing
    lines = text.split('\n')
    amounts = []
    
    # Priority patterns - check for final amounts first
    priority_patterns = [
        # High priority: Amount payable, Final amount, Net amount
        r'(?:Amount payable|Net amount|Final amount|Amount due|Total payable|Payable amount)[:\s-]*(\d{1,6}(?:\.\d{0,2})?)',
        # Total with various formats
        r'(?:Total charge|Total amount|Grand total|Final total)[:\s-]*(\d{1,6}(?:\.\d{0,2})?)',
        # Amount after discount
        r'(?:After discount|Post discount)[:\s-]*(\d{1,6}(?:\.\d{0,2})?)'
    ]
    
    # Check priority patterns first
    for pattern in priority_patterns:
        for line in lines:
            try:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        amount_val = float(str(match).strip())
                        if amount_val >= 1.0:
                            print(f"Priority match found: {amount_val} from line: {line.strip()}")
                            amounts.append(amount_val)
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                print(f"Priority pattern error: {str(e)}")
                continue
    
    # If priority patterns found amounts, return the highest one
    if amounts:
        final_amount = max(amounts)
        print(f"Using priority amount: {final_amount}")
        return final_amount
    
    # Fallback patterns for other amount formats
    fallback_patterns = [
        # Pattern for "Total: $XXX.XX" or "Amount: $XXX.XX"
        r'(?:Total|Amount|Bill|Charge)[:\s]*[\$₹€£]?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{0,2})?)',
        
        # Pattern for standalone currency amounts
        r'[\$₹€£]\s*(\d{1,6}(?:,\d{3})*(?:\.\d{0,2})?)',
        
        # Pattern for amounts at end of lines (common in bills)
        r'(\d{1,6}(?:\.\d{2}))\s*$',
        
        # Pattern for "XXX.XX USD" or similar
        r'(\d{1,6}(?:,\d{3})*(?:\.\d{0,2}))\s*(?:USD|INR|EUR|GBP)',
        
        # Pattern for amounts after dash or colon
        r'[-:]\s*(\d{1,6}(?:\.\d{0,2})?)\s*$',
        
        # Pattern for decimal amounts without currency symbols
        r'\b(\d{3,6}(?:\.\d{2})?)\b'
    ]
    
    for pattern in fallback_patterns:
        try:
            # Search line by line for context-aware extraction
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['total', 'amount', 'due', 'bill', 'charge', 'pay', 'fee']):
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        try:
                            clean_amount = str(match).replace(',', '').strip()
                            amount_val = float(clean_amount)
                            if amount_val >= 10.0:  # Reasonable minimum for medical bills
                                print(f"Fallback match found: {amount_val} from line: {line.strip()}")
                                amounts.append(amount_val)
                        except (ValueError, TypeError):
                            continue
                            
        except Exception as e:
            print(f"Fallback pattern error: {str(e)}")
            continue
    
    if amounts:
        # Remove duplicates and sort
        unique_amounts = list(set(amounts))
        unique_amounts.sort(reverse=True)
        
        # Filter out unrealistic amounts
        reasonable_amounts = [amt for amt in unique_amounts if 10.0 <= amt <= 1000000]
        
        if reasonable_amounts:
            final_amount = reasonable_amounts[0]
            print(f"Final extracted amount: {final_amount}")
            return final_amount
    
    print("No amount found")
    return None

def extract_diagnosis_with_regex(text):
    """Enhanced regex to find diagnosis with more patterns"""
    import re
    
    patterns = [
        r'Diagnosis[:\s]+(.*?)(?:\n|$|\.)',
        r'Condition[:\s]+(.*?)(?:\n|$|\.)',
        r'Reason for Visit[:\s]+(.*?)(?:\n|$|\.)',
        r'Treatment for[:\s]+(.*?)(?:\n|$|\.)',
        r'Presenting Complaint[:\s]+(.*?)(?:\n|$|\.)',
        r'Chief Complaint[:\s]+(.*?)(?:\n|$|\.)',
        r'Primary Diagnosis[:\s]+(.*?)(?:\n|$|\.)',
        r'Medical Condition[:\s]+(.*?)(?:\n|$|\.)',
        r'Disease[:\s]+(.*?)(?:\n|$|\.)',
        r'Illness[:\s]+(.*?)(?:\n|$|\.)'
    ]
    
    for pattern in patterns:
        try:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                diagnosis = match.group(1).strip()
                # Clean up the diagnosis text
                diagnosis = re.sub(r'\s+', ' ', diagnosis)  # Replace multiple spaces
                if len(diagnosis) > 3 and len(diagnosis) < 100:  # Reasonable length
                    return diagnosis
        except Exception as e:
            print(f"Diagnosis extraction error: {str(e)}")
            continue
    
    # Fallback: Look for common medical terms
    medical_terms_pattern = r'\b(pregnancy|fever|cancer|diabetes|fracture|injury|infection|bodyache|headache|asthma|flu|cold|pneumonia|bronchitis|hypertension|migraine|arthritis|gastritis|appendicitis|tonsillitis|sinusitis|dermatitis|conjunctivitis)\b'
    try:
        match = re.search(medical_terms_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).title()
    except Exception as e:
        print(f"Medical terms extraction error: {str(e)}")
    
    return None

def clean_and_convert_amount(amount):
    """Enhanced amount conversion with better error handling"""
    if amount is None:
        return 0.0
        
    if isinstance(amount, str):
        # Remove currency symbols, commas, and extra spaces
        clean_amount = re.sub(r'[^\d.]', '', amount.strip())
        try:
            if clean_amount:
                return float(clean_amount)
            else:
                return 0.0
        except ValueError:
            return 0.0
    elif isinstance(amount, (int, float)):
        return float(amount)
    else:
        return 0.0

def get_bill_info(data):
    """Enhanced bill information extraction with better fallbacks"""
    print(f"Processing bill text (first 200 chars): {data[:200]}...")
    
    # First try to extract with enhanced regex
    diagnosis = extract_diagnosis_with_regex(data)
    expense = extract_expense_with_regex(data)
    
    print(f"Regex extraction - Diagnosis: {diagnosis}, Expense: {expense}")
    
    if diagnosis and expense is not None and expense > 0:
        return {'disease': diagnosis, 'expense': expense}
    
    # If regex fails, try Hugging Face API
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    prompt = f"""From this medical bill text, extract:
1. The medical condition or disease being treated
2. The total amount or expense (as a number only)

Text: {data[:1000]}

Return in this exact JSON format: {{"disease":"condition name","expense":"amount as number"}}"""
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result and isinstance(result, list) and 'generated_text' in result[0]:
            response_text = result[0]['generated_text']
            print(f"API response: {response_text}")
            
            try:
                # Try to parse as JSON
                parsed_result = json.loads(response_text)
                api_diagnosis = parsed_result.get('disease', '')
                api_expense = clean_and_convert_amount(parsed_result.get('expense', 0))
                
                # Use API results if they're better than regex
                final_diagnosis = api_diagnosis if api_diagnosis and len(api_diagnosis) > 2 else diagnosis
                final_expense = api_expense if api_expense > 0 else expense
                
                if final_diagnosis and final_expense and final_expense > 0:
                    return {'disease': final_diagnosis, 'expense': final_expense}
                    
            except json.JSONDecodeError:
                # Try to extract JSON from text using regex
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    try:
                        parsed_result = json.loads(json_match.group(0))
                        api_diagnosis = parsed_result.get('disease', '')
                        api_expense = clean_and_convert_amount(parsed_result.get('expense', 0))
                        
                        final_diagnosis = api_diagnosis if api_diagnosis else diagnosis
                        final_expense = api_expense if api_expense > 0 else expense
                        
                        if final_diagnosis and final_expense and final_expense > 0:
                            return {'disease': final_diagnosis, 'expense': final_expense}
                    except json.JSONDecodeError:
                        pass
                        
    except requests.exceptions.RequestException as e:
        print(f"Hugging Face API error: {str(e)}")
    except Exception as e:
        print(f"API processing error: {str(e)}")
    
    # Fallback to local model if available
    if local_qa:
        try:
            disease_result = local_qa(question="What is the primary medical condition or disease being treated?", context=data)
            expense_result = local_qa(question="What is the total expense amount on this medical bill?", context=data)
            
            local_disease = disease_result['answer'] if disease_result['score'] > 0.1 else None
            local_expense = clean_and_convert_amount(expense_result['answer'])
            
            final_diagnosis = local_disease if local_disease else diagnosis
            final_expense = local_expense if local_expense > 0 else expense
            
            if final_diagnosis and final_expense and final_expense > 0:
                return {'disease': final_diagnosis, 'expense': final_expense}
                
        except Exception as e:
            print(f"Local model error: {str(e)}")
    
    # Final fallback - use whatever we found
    final_diagnosis = diagnosis if diagnosis else "See Claim Reason"
    final_expense = expense if expense and expense > 0 else 0.0
    
    return {'disease': final_diagnosis, 'expense': final_expense}

def is_disease_excluded(disease, exclusion_list):
    """Enhanced disease exclusion check with better matching"""
    if not disease:
        return False
        
    disease_lower = disease.lower().strip()
    
    # Don't exclude if disease is just a placeholder
    if disease_lower in ['see claim reason', 'extraction failed', 'not found', '']:
        return False
        
    # Check for exact and partial matches
    for exclusion in exclusion_list:
        exclusion_lower = exclusion.lower().strip()
        
        # Check for exact match or if exclusion is contained in disease
        if exclusion_lower in disease_lower or disease_lower in exclusion_lower:
            # Additional check for common false positives
            if exclusion_lower == "pregnancy" and "pregnancy test" in disease_lower:
                continue  # Pregnancy test is not the same as pregnancy treatment
            return True
            
    return False

def generate_claim_report(patient_info, bill_info, claim_amount_str):
    """Enhanced claim report generation with better validation logic"""
    
    # Convert claim amount to float with better error handling
    try:
        claim_amount = float(str(claim_amount_str).replace(',', '').replace('$', '').strip())
    except (ValueError, AttributeError):
        claim_amount = 0.0

    # Get and clean bill expense
    bill_expense = clean_and_convert_amount(bill_info.get('expense'))
    
    # Get disease from bill or fallback to claim reason
    disease = bill_info.get('disease', patient_info.get('claim_reason', ''))
    
    print(f"Claim validation - Claim Amount: {claim_amount}, Bill Expense: {bill_expense}, Disease: {disease}")
    
    # Enhanced validation checks
    info_complete = all([
        patient_info.get('name', '').strip(),
        patient_info.get('address', '').strip(),
        patient_info.get('claim_reason', '').strip(),
        claim_amount > 0,  # Claim amount should be positive
        bill_expense > 0   # Bill expense should be positive
    ])
    
    excluded = is_disease_excluded(disease, general_exclusion_list)
    
    # Enhanced decision logic with tolerance for minor differences
    amount_tolerance = max(1.0, bill_expense * 0.01)  # Allow 1% tolerance or minimum $1
    amount_difference = abs(claim_amount - bill_expense)
    amounts_match = amount_difference <= amount_tolerance
    claim_within_bill = claim_amount <= (bill_expense + amount_tolerance)
    
    # Decision logic
    if not info_complete:
        missing_fields = []
        if not patient_info.get('name', '').strip():
            missing_fields.append('name')
        if not patient_info.get('address', '').strip():
            missing_fields.append('address')
        if not patient_info.get('claim_reason', '').strip():
            missing_fields.append('claim reason')
        if claim_amount <= 0:
            missing_fields.append('valid claim amount')
        if bill_expense <= 0:
            missing_fields.append('valid bill amount')
            
        decision = f"REJECTED: Incomplete information (missing: {', '.join(missing_fields)})"
        
    elif excluded:
        decision = f"REJECTED: Treatment for '{disease}' is excluded under policy terms"
        
    elif not claim_within_bill:
        excess_amount = claim_amount - bill_expense
        decision = f"REJECTED: Claim amount (${claim_amount:.2f}) exceeds bill amount (${bill_expense:.2f}) by ${excess_amount:.2f}"
        
    else:
        # Approved cases
        if amounts_match:
            decision = f"APPROVED: Full claim amount of ${claim_amount:.2f}"
        else:
            # Approve up to the bill amount if claim is higher
            approved_amount = min(claim_amount, bill_expense)
            decision = f"APPROVED: ${approved_amount:.2f} (claim: ${claim_amount:.2f}, bill: ${bill_expense:.2f})"
    
    # Generate comprehensive report
    report = f"""
    CLAIM DECISION: {decision}
    
    Executive Summary
    This report details the analysis of the insurance claim submitted by {patient_info['name']}. 
    The claim has been evaluated based on completeness of information, policy exclusions, and amount validation.
    
    Claim Details
    - Patient: {patient_info['name']}
    - Claim Reason: {patient_info['claim_reason']}
    - Medical Facility: {patient_info['medical_facility']}
    - Date of Service: {patient_info['date']}
    - Claim Amount: ${claim_amount:.2f}
    - Bill Amount: ${bill_expense:.2f}
    - Detected Condition: {disease}
    - Amount Difference: ${abs(claim_amount - bill_expense):.2f}
    
    Verification Results
    - Information Complete: {'Yes' if info_complete else 'No'}
    - Covered Condition: {'No' if excluded else 'Yes'}
    - Amount Valid: {'Yes' if claim_within_bill else 'No'}
    - Amounts Match: {'Yes' if amounts_match else 'No'}
    
    Final Decision
    {decision}
    
    Additional Notes
    - Claim processing used enhanced validation with tolerance for minor amount differences
    - Bill text extraction: {'Successful' if bill_expense > 0 else 'Failed - manual review recommended'}
    """
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_claim():
    # Extract form data
    patient_info = {
        'name': request.form.get('name', '').strip(),
        'address': request.form.get('address', '').strip(),
        'claim_type': request.form.get('claim_type', '').strip(),
        'claim_reason': request.form.get('claim_reason', '').strip(),
        'date': request.form.get('date', '').strip(),
        'medical_facility': request.form.get('medical_facility', '').strip(),
        'total_claim_amount': request.form.get('total_claim_amount', '').strip(),
        'description': request.form.get('description', '').strip()
    }
    
    medical_bill = request.files.get('medical_bill')
    
    if not medical_bill or medical_bill.filename == '':
        return render_template("result.html", 
                              output="No medical bill uploaded. Please upload a valid PDF file.",
                              **patient_info)
    
    # Process bill
    bill_text = get_file_content(medical_bill)
    if not bill_text.strip():
        return render_template("result.html", 
                              output="The uploaded bill is empty or could not be read. Please ensure the PDF contains readable text.",
                              **patient_info)
    
    print(f"Extracted bill text length: {len(bill_text)}")
    
    bill_info = get_bill_info(bill_text)
    
    print(f"Final bill info: {bill_info}")
    
    # Handle expense extraction failures more gracefully
    if bill_info.get('expense') is None or bill_info.get('expense') <= 0:
        return render_template("result.html", 
                              output=f"Could not extract expense amount from bill. Bill info extracted: {bill_info}. Please resubmit with clearer documentation or check if the PDF contains readable text.",
                              **patient_info)
    
    # Generate claim report
    report = generate_claim_report(patient_info, bill_info, patient_info['total_claim_amount'])
    
    # Format for HTML display
    formatted_report = report.replace('\n', '<br>')
    
    return render_template("result.html", 
                          output=formatted_report,
                          **patient_info)

def test_extraction():
    """Test function for debugging the specific PDF case"""
    test_text = """APOLLO HOSPITALS
Patient Information:
- Name: Surbhit
- Date of Birth: 01/01/1223
- Address: India
- Phone Number:1239874653
Service Details:
Date of Service: 02/02/1233
Diagnosis: Bodyache with fever, cold and could
Details: Continue having the given medicines as mentioned in the prescription
Service charges:
Doctor's fee - 1500
Medicines - 2000
Total charge - 3500
Membership Discount - 10%
Amount payable - 3150"""
    
    print("=== TESTING EXTRACTION ===")
    diagnosis = extract_diagnosis_with_regex(test_text)
    expense = extract_expense_with_regex(test_text)
    print(f"Extracted Diagnosis: {diagnosis}")
    print(f"Extracted Expense: {expense}")
    print("=========================")
    
    return {'disease': diagnosis, 'expense': expense}

if __name__ == '__main__':
    # Run test before starting the app
    test_extraction()
    app.run(host='0.0.0.0', port=8081, debug=True)

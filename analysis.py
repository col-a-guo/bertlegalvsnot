import pandas as pd
import matplotlib.pyplot as plt
import io

# Read the data from the provided text file
# Since the data is in CSV format inside a text file, we'll first parse it
def analyze_case_outcomes(text_data):
    # Use pandas to read the CSV data
    df = pd.read_csv(io.StringIO(text_data))
    
    # Check the balance of case_outcome column
    outcome_counts = df['case_outcome'].value_counts()
    outcome_percentages = df['case_outcome'].value_counts(normalize=True) * 100
    
    # Combine counts and percentages into a dataframe for better display
    balance_df = pd.DataFrame({
        'Count': outcome_counts,
        'Percentage': outcome_percentages.round(2)
    })
    
    # Print the balance information
    print("Balance of case_outcome values:")
    print(balance_df)
    print("\nTotal number of cases:", len(df))
    
    # Create a pie chart to visualize the balance
    plt.figure(figsize=(10, 6))
    plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Case Outcomes')
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('case_outcome_distribution.png')
    plt.close()
    
    print("\nPie chart saved as 'case_outcome_distribution.png'")
    
    return balance_df

# Extract the CSV data from the text content
def extract_csv_from_text(text_content):
    # Skip the first line if it's not part of the CSV data
    lines = text_content.split('\n')
    if "case_id,case_outcome,case_title,case_text" in lines[0]:
        # The CSV data starts from the first line
        return text_content
    
    # Find the line that starts the CSV data
    for i, line in enumerate(lines):
        if "case_id,case_outcome,case_title,case_text" in line:
            return '\n'.join(lines[i:])
    
    return None

# Main function to execute the analysis
def main():
    # Read the text file
    try:
        with open('legal_text_classification.csv', 'r', encoding='utf-8') as file:
            text_content = file.read()
        
        # Extract the CSV data
        csv_data = extract_csv_from_text(text_content)
        
        if csv_data:
            # Analyze the data
            analyze_case_outcomes(csv_data)
        else:
            print("Could not find CSV data in the provided text file.")
    
    except FileNotFoundError:
        print("Error: 'paste.txt' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
# BigFlowPrototypes

A package for preprocessing data for AI models.

## Modules

### Image.py

Processes image data using VGG16 to convert it to vector data and store it in a database.

### linearEmbed.py

Prepares a "normal dataset" and converts non-numeric columns to embedded data using OpenAI embedding.

### Linear.py

Removes non-numeric data and inputs it into a model.

## Installation

```bash
pip install BigFlowPrototypes


from data_preprocessor import linearEmbed

if __name__ == "__main__":
    input_data_path = 'titanic.csv'  # Path to the input data file
    label_column = 'Survived'  # Name of the label column
    columns_to_embed = ['Name']  # Columns to embed
    openai_api_key = 'your_openai_api_key'  # OpenAI API key
    
    X_train, X_test, y_train, y_test = linearEmbed.main(input_data_path, label_column, columns_to_embed, openai_api_key)
    print("Data processing complete.")


from data_preprocessor import Linear

if __name__ == "__main__":
    input_file = 'movies.csv'  # Path to the input data file
    label = 'Popularity'  # Name of the label column
    
    X_train, X_test, y_train, y_test = Linear.main(input_file, label)
    print("Data processing complete.")


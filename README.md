# Neural Network Charity Analysis

> Creating a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Overview of Project

Preprocessing the data for the neural network, Compile,train and evaluate the model & finally optimizing the model.

## Results

### Data Preprocessing
After looking at the data, I established that the target variable is the "IS_SUCCESSFUL" column. I then removed the "EIN" and "NAME" columns as they did not offer any relevant data that could help the model perform better. The remaining columns became the features for the model.

### Compiling, Training, and Evaluating the Model
For my first attempt at compiling a neuron network consisted of 8 neurons in the first layer and 6 in the second. Both layers had relu activation functions and the output layer had a sigmoid activation function. I started with these parameters as relu does better with nonlinear data, and two layers allows for a second layer to reweight the inputs from the first layer. Here are the preformance metrics of this model.

### Charts

## Summary



## Todo Checklist

A helpful checklist to gauge how your README is coming on what I would like to finish:

- [ ] Improve the Data Processing.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

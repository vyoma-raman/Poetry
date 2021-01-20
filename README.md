# Poetry Generation

## Description
This is a Python Tensorflow package for poetry generation.

The corpus is extracted from author pages in the *Daily Californian*. It may be modified to work with other sources in the `getCorpus` method of the `newGenerator` class in `generators.py`.

## Table of Contents
* [Dependencies](#dependencies)
* [Usage](#usage)
* [Credits](#credits)

## Dependencies

To use this Python package, make sure the following libraries are installed.

**pickle5**

[Installation](https://pypi.org/project/pickle5/)
```
pip install pickle5
```

**TensorFlow**

[Installation](https://www.tensorflow.org/install)
```
pip install tensorflow
```

**NLTK**

[Installation](https://www.nltk.org/install.html)
```
pip install nltk
```

**Requests**

[Installation](https://requests.readthedocs.io/en/master/user/install/)
```
pip install requests
```

**NumPy**

[Installation](https://numpy.org/install/)

\*`conda` installation recommended
```
pip install numpy
```

## Usage
Navigate to the `scripts` folder of the package, where a generator can be loaded using your terminal of choice.
### Creating a New Generator
Run the following command:
```
python3 main.py new
```
Upon prompting, type the id of the author to scrape from (normally first initial + last name), the number of pages of work in their *Daily Californian* URL, and the desired accuracy of the model.

For example: [Vyoma Raman](https://www.dailycal.org/author/vraman)
```
Author name for new generator: vraman
Number of pages of content from this author: 2
Enter desired accuracy: .8
```
If desired, save the generator.
```
Save new generator? (y/n): y
```
### Using an Existing Generator
Run the following command:
```
python3 main.py existing
```
Upon prompting, type the id of the author whose work as used to train the model and the accuracy of the model.
```
Author name for existing generator:  vraman
Accuracy of existing generator: .8
```
### Generating Poetry
Upon prompting, enter a few words of seed text to generate the poem with and the number of words to generate.
```
Type text to start the poem: Hello from the other side
Type additional word length of the poem: 50
```
The generated poem will then display.
```
hello from the other side of the only way i can by eating pizza at the same time and ponchos or rain jackets pose the same problem to me as sweatshirts who don't have to plan for a delicate way to extract myself if i cannot get into a lift if i cannot be taken
```
If desired, save the poem. It will be saved to a text file in the `poems` folder which can be edited by hand.
```
Save this poem? (y/n): y
```
If desired, generate another poem using the same model.
```
Generate another poem? (y/n): y
```
## Credits
This package was built by Vyoma Raman. It was inspired by the poetry model built in [Natural Language Processing in TensorFlow](https://www.coursera.org/learn/natural-language-processing-tensorflow) offered by DeepLearning.AI on [Coursera](https://www.coursera.org/).
import util
import sys
from generators import existingGenerator, newGenerator

if sys.argv[1] == "new":
    author_name = input("Author name for new generator: ")
    num_pages = int(input("Number of pages of content from this author: "))
    accuracy = str(float(input("Enter desired accuracy: ")))
    g = newGenerator(author_name, num_pages, accuracy)
    save_gen = input("Save new generator? (y/n): ")
    if save_gen == "y":
        util.saveGenerator(g)
elif sys.argv[1] == "existing":
    author_name = input("Author name for existing generator: ")
    accuracy = str(float(input("Accuracy of existing generator: ")))
    g = util.loadGenerator(author_name, accuracy)
else:
    sys.exit("Please indicate whether you would like to use a new or existing generator.")
generate_another = True
while generate_another:
    seed_text = input("Type text to start the poem: ").lower()
    length = int(input("Type additional word length of the poem: "))
    poem = g.generatePoem(seed_text, length)
    print(poem)
    save_poem = input("Save this poem? (y/n): ")
    if save_poem == "y":
        util.savePoem(poem, author_name, accuracy, seed_text)
    generate_another = input("Generate another poem? (y/n): ") == "y"
# Jarvis

## Set-Up

You should first download GoogleNews-vectors-negative300 as a binary file.

You can follow this link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing <br>
OR <br>
Download from this repo: https://github.com/mmihaltz/word2vec-GoogleNews-vectors/tree/master <br>

Then, change the file path inside the script.

## Used Technologies

### Gensim:

Used for preprocessing prompts and word vectoring.

### Scikit-Learn:

Used to identify similarities to choose text from the dataset.

### Numpy and Pandas:

Used to structure data and create vectors.

## Features

### Finding the most related data with the prompt

This script allows user to enter a prompt to find a related text in the dataset. For example, if user wants to learn something it can directly be asked.

### Example input:</br>
"What will happen in the following month to the company?"</br>
### Output:</br>
Data 27: Tech talk on cybersecurity best practices is scheduled for the first Friday of next month.
Data 23: The new feature set for the mobile app will be discussed in the next product team meeting.
Data 18: The annual performance review schedule will be shared by HR by the end of the week.

Jester Data: These data are approximately 1.7 million ratings in the range
[1,2,3,4,5] of 150 jokes from 63,974 users. These data are from the Eigentaste
Project at Berkeley. We've munged the data somewhat, so use the local copies
here: 

== For problem 2 == 

ratings.dat.gz: Each row is formatted as [User ID] [Joke ID] [Rating]

== For problem 3 == 

jester_items.clean.dat: Maps the joke IDs to the joke text. (No punctuation,
lowercased.)

generate_binary_features.py: Maps the previous file into a collection of binary
features which will be stored in the file "X.txt".


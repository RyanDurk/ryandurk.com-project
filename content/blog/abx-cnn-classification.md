---
title: "Classifying antibiotic skeletal structures with a convolutional neural network"
date: 2019-04-08T09:10:15-05:00
draft: false
---
The first task of [fast.ai](https://www.fast.ai) is training a convolutional neuro network to classify images. We'll group antibiotics by their skeletal structure. The aim will be doing this as rapidly as possible to gain a working understanding of the concepts. We'll acknowledge some shortcomings once we see our model's failures. The data will be from [PubChem's Classification Browser](https://pubchem.ncbi.nlm.nih.gov/classification/#hid=1) and we'll use [the fast.ai library](https://docs.fast.ai/), [ResNet](https://arxiv.org/abs/1512.03385) and [Google Colab](https://colab.research.google.com/).
## The classes
[RM Coates et all](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3085877/) nicely lists antibiotic classes that will form our search on PubChem with some changes.\* Ontologically speaking	&beta;-lactams encompass carbapenems and monobactams but those are listed as their own headers. We'll add penicillins and cephalosporins and remove &beta;-lactams as a classification. Ketolides are a subset of macrolides. The US spelling, and therefore MEsH spelling of sulphonamide is actually "sulfanilamide," so we'll make that change as well.

\* The author found the list of antibiotics first and then MeSH. Since "Anti-Bacterial Agents" is a MeSH term, a more straightforward approach may be choosing categories directly from the [MEsH browser](https://meshb.nlm.nih.gov/record/ui?name=Anti-Bacterial%20Agents).

Our antibiotic classes:

* Pencillins
* Cephalosporins
* &beta;-Lactamase inhibitors
* Carbapenems
* Monobactams
* Aminoglycosides
* Tetracyclines
* Rifamycins
* Macrolides
* Lincosamides
* Glycopeptides
* Lipopeptides
* Streptogramins
* Sulfanilamide
* Oxazolidinones
* Quinolones

Searching PubChem by the class name gives us antibiotics in each group with some tweaks. Replace '&beta;' with 'beta', for instance. You'll find an option to download the entire MeSH grouping images at once. Extract those into individual directories named by their class name. The filenames should be something like ./aminoglycoside/816.png, where the number (816 in this case) is the PubChem CID. The CID is unique for each molecule.
## Data munging
Antibiotics can be members of multiple classes. Let's find which ones are.

```duplicates.py
import os
import re
from pathlib import Path

def duplicates_among_set_dict(set_dict):
    """Return a dict[keys] = values such that keys are duplicate set values and values
        are duplicate keys

    >>> no_dups = no_dups = {'a': {1}, 'b': {2}, 'c': {3}}
    >>> some_dups = {'a': {1, 2, 3}, 'b': {3, 4, 5}, 'c': {3, 5, 6}, 'd': {1, 2}}
    >>> duplicates_among_set_dict(no_dups)
    {}
    >>> duplicates_among_set_dict(some_dups)
    {1: ['a', 'd'], 2: ['a', 'd'], 3: ['a', 'b', 'c'], 5: ['b', 'c']}
    """

    keylist = list(set_dict.keys())

    duplicates = dict()

    for i in range(len(keylist)):
        primary_key = keylist[i]

        for value in set_dict[primary_key]:
            for j in range(i+1, len(keylist)):
                secondary_key = keylist[j]

                if value in set_dict[secondary_key]:
                    if value in duplicates:
                        if secondary_key not in duplicates[value]:
                            duplicates[value].append(secondary_key)
                    else:
                        duplicates[value] = [primary_key, secondary_key]
    
    return duplicates

data_directory = Path('abx')
class_dirs = os.listdir(data_directory)
abx = dict()
for class_dir in class_dirs:
    abx[str(class_dir)] = set(os.listdir(data_directory/class_dir))

duplicates = duplicates_among_set_dict(abx)
```

We may need duplicates\_among\_set\_dict's functionality in a future model, so it's made into a function. There are a little over 100 duplicates, most belonging to just 2 groups. Not surprisingly, many &beta;-lactamase inhibitors fall under pencillins as &beta;-lactamase obviously binds both &beta;-lactams and &beta;-lactamase inhibitors-- they should have similar structures. For this group, we'll remove the penicillin classification since they are primarily clinically useful as &beta;-lactamases.
```duplicates.py
for dup, dup_classes in duplicates.items():
    if 'beta-lactamase inhibitors' in dup_classes:
        for dup_class in dup_classes:
            if dup_class != 'beta-lactamase inhibitors':
                os.remove(f'abx/{dup_class}/{dup}')
```
That leaves 79 duplicate classifications. For those, the author went through PubChem's description page to manually assign what was felt to be the most clinically relevant class. These were saved in resolution.csv for reproducibility.
```duplicates.py
resolution_file = 'resolution.csv'
if not os.path.isfile(resolution_file):
    with open(resolution_file, 'w') as fout:
        for dup in duplicates:
            cid = re.match(r'(\d+)\.png', dup)
            cid = cid.group(1)

            print(f"https://pubchem.ncbi.nlm.nih.gov/{cid}")
            while True:
                cl = input('What class? ')

                if cl in os.listdir('abx'):
                    break
                
                if cl == 'NULL':
                    break
            
            fout.write(f'{cid},{cl}\n')
else:
    with open(resolution_file) as fin:
        for line in fin:
            line = line.strip()
            cid, correct_class= line.split(',')

            if cid in duplicates:
                for dup_class in duplicates[cid+'.png']:
                    if dup_class != correct_class:
                        os.remove(f'abx/{dup_class}/{cid}.png')
```

Upon individual inspection, perhaps 2/3 of the classifications were easily resolved and the rest were irrelevant to our model and therefore removed. For instance, some CIDs refer to a mixture of antibiotics, which becomes an issue when multiple ones from different classes are in the same picture.

## The model
We'll start with these lines taken from the fast.ai Lecture 1 notebook--
```model.py
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.metrics import error_rate
bs = 64
```

Our first change is simply pointing path to wherever we're storing the data files. The author uploaded them to Google Drive. You could also tar them and put them online, using the untar function included in the fast ai library. Unfortunately you can't simply scp into a Google Colab notebook. pat is a regex that when applied to the image path gives the ImageDataBunch.from_name_re factory method a grouping with the class name of each image:
```model.py
from google.colab import drive
drive.mount('/content/gdrive')

from pathlib import Path
import os

path = Path('/content/gdrive/My Drive/abx')
fnames = []
for abx_path in path.ls():
  if os.path.split(abx_path)[1] not in ['models']: #exclude these directories
    fnames.extend(abx_path.ls())
pat = r'/([^/]+)/\d+.png$'
```
Below, np.random.seed(2) is unexplained in the fast.ai lecture. It may help ImageDataBunch to produce reproducible learning and training sets. 224 is a magic number, of which we're told will be explained in a later lecture. It is the length of a square such that 224 = 7 * 2**n. get_transforms() resizes our images to 224x224 squares, in keeping with that magic length. bs is a batchsize, 64 works well enough for our colab GPU. I'm not sure if normalization adds much here considering the images already have consistent, high-contrast features, but we'll do that anyway and evaluate the resulting images.
```model.py
np.random.seed(2)
data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
```
Calling show_batch on our ImageDataBunch object gives us the below image. We see 9 structures, each correctly classified and entirely contained in the transformaed image. Far from perfect, these post-transformation images have artifacts making it difficult to see larger structures. We also have some extraneous information in these images, like salts and even other large molecules. Those could be filtered programmatically, or by using another image source (PubChem stores many mixtures of images). But, looking at the pencillin for instance, we can clearly see a &beta;-lactam ring with its requisite 4-member square shape containing an amide group (red oxygen, carbon, blue nitrogen motif). So, there's probably enough information in these images for proper classification. In all, we could do some data-cleaning and find better images (ideally we'd just create 224x224 square images exactly instead of downloading high-quality ones from PubChem and then shrinking those). But we're aiming for proof-of-concept more than, say, creating a production model, so let's continue as is.
{{< figure src="/lec_1_show_batch.png" alt="data.show_batch" class="center_text" >}}
```model.py
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
```
We're using ResNet 34. Obviosuly we could use 50 or higher, but we'll stick to our goal of proof-of-concept.

|epoch|	train\_loss|	valid\_loss|	error\_rate|	time|
|---------|--------|----------|------------|----------|
|0|	1.456029	|0.625639	|0.187248	|23:28|
|1|	0.790583	|0.462501	|0.137207	|01:13|
|2|	0.562697	|0.409061	|0.125908	|01:10|
|3|	0.484117	|0.402371	|0.122680	|01:09|


Ok, pretty good. The default learning rate and loss function gave us a continously decreasing error\_rate after 4 runs. The first run took 20x the time as subsequent ones.
Let's look at our top-losses with the [Grad-CAM heatmap](http://gradcam.cloudcv.org/).
```model.py
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11), heatmap=True)
```
{{< figure src="/lec_1_top_losses.png" alt="Top losses after 1 iteration" class="center_text" >}}
In four of the nine images the model seems unduly concerned with whitespace. It's unsurprising to see classification failures in those cases. On the other hand, the misclassified macrolide-aminoglycoside pairings look like they actually belong to both classes. And the calculated worst loss in the top left corner is actually correctly classified by our model as an aminoglycoside (our test data is incorrect in that case). Similarly, the 8th top loss (row 3, col 2) appears to be correctly classified by the mode. We see some trouble with tetracycline classification in two cases. The quinolone/macrolide molecule in the bottom-right has a bizarre structure, another unsurprising misclassification. It's hard to say, at this point, how much a role of orientation and positioning have on our model's prediction-- could we be overfitting based on that? Perhaps better training/test sets would be randomized.

We've got many things we could test to improve with the input data at this point. It pains me to ignore them and keep moving, but we're going to stick to our goal of showing feasibility and not perfection. Let's refine the model a little bit, and finish things there so we can move on to the next [fast.ai](https://www.fast.ai) lecture.
```model.py
learn.lr_find()
learn.recorder.plot()
```
{{< figure src="/lec_1_lr.png" alt="Learning rate loss graph" class="center_text" >}}
Looks like our loss picks up with learning rates above 1e-3. So let's do a couple more runs with a max LR of 1e-6 to 1e-4. This happens to be the same learning rate chosen by Jeremy Howard during the fast.ai lecture, in which he also started with ResNet-34 to train image data.
```model.py
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
```
|epoch|	train\_loss|	valid\_loss|	error\_rate|	time|
|---------|--------|----------|------------|----------|
|0	|0.281094	|0.270655	|0.073446	|01:15|
|1|	0.259531	|0.262721	|0.072639	|01:15|

That's an improvement in error rate from before, but it looks like we're leveling out at 7.2ish%. How do our top losses look now?
```model.py
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11), heatmap=True)
```
{{< figure src="/lec_1_top_losses_2nd.png" alt="Final top losses" class="center_text" >}}
Much better. Again, our worst loss is actually correctly classified by our model and just misclassified in the source data, as are the 4th (row 2, col 1) and 5th (row 2, col 2) top losses. Our heatmaps in the rest fit the molecules exactly-- quite an improvement. We'll stop here, and move on to the next fast.ai lecture.
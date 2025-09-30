# WoMAD Project Logs

This page includes an overview and all the logs for the WoMAD project.

* Get a glance at the process (and hopefully, progess) of WoMAD: [Chronological Logs](#chronological-logs)
* Take a look at our upcoming plans for the model: [Upcoming Versions and Plans](upcoming.md)

---

## Overview:

* Current version of WoMAD: 0.1.4
* Current state: Basic structure of modules is defined through simple function and docstrings
* Upcoming: Populating the modules with code, sampling the  dataset (10 subjects) for basic pipeline tests, and creation of a detailed outline for the documenation

---

## Chronological Logs

### WoMAD v0.1.0

#### Log #1: 2025-09-07

**Created the Deploy Docs Workflow (v0.1.1)**

Although the code is basically thin air at this point, Saameh decided it would be a good idea to *finalize* the structure of the repository. So, we got our fist error. Shocking! We know!

Hence, v0.1.1 was created. A mostly empty repository that has a decent, but mostly empty documentation on [our GitHub page](https://saamehsanaaee.github.io/WoMAD)

#### Log #2: 2025-09-09

**Expanded the Documentation structure (v0.1.2)**

Do you see a pattern?

Before outlining the actual `.py` files, *someone* decided it would be a good idea to make the document and the GitHub page for our lovely WoMAD model *better*. What is "better"? Nobody knows.

Our next tasks are:

* Outlining all `.py` files and complete their description docstrings
* Outlining `the-code.md` and `the-science.md` files
* Expand main `README.md` file

#### Log #3: 2025-09-10

**Filled docstrings of the `.py` files to describe each module (v0.1.2)**

The `.py` files are now partially outlined on a very high level. The next step is to make this outline more granular and create the logical flow of the code within each module.

#### Log #4: 2025-09-18

**Outlined all `.py` files to create the basic structure of the modules (v0.1.3)**

The `.py` files are now properly outlined. The next step is to populate each section with their respective functions.

#### Log #5: 2025-09-30

**Created basic structure for all WoMAD modules (v0.1.4)**

All WoMAD `.py` module files now have the basic structure of the code. Functions, docstrings, paths, etc. have been added and the modules now require a "fill in the blanks" step to create the first iteration of the WoMAD model, ready for basic tests.

Here's the exact list of changes made in the commits for v0.1.4:

* `WoMAD_config.py` included improper and lengthy variable names. They are improved now.
* All modules inside `./WoMAD/` directory included dashes in the names instead of underscores. This is edited to ensure proper import of the modules when necessary.
* All `.py` files now include the basic code structure that ensure the logical flow we need the data to go through. Thismeans that after the current version, if no bugs are found (LOL, right?), we can move on to create the actual code for each module which you can see in [the upcoming page](upcoming.md).
* Other minute changes have been made in docs, the `.toml` file, and other "housekeeping" parts of the repository that can be called "small edits" for now. (No mojor changes have been made in the non-`.py` files.)

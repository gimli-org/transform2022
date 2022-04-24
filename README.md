# Geophysical modeling & inversion with pyGIMLi

<div width="50%">
<img src="https://www.pygimli.org/_images/pg_logo.png" width=40%>
</div>

The [pyGIMLi tutorial at Transform 22](https://transform.softwareunderground.org/overview/)

Instructors:
[Florian Wagner](https://github.com/florian-wagner) <sup>1</sup>, [Carsten Rücker](https://github.com/carsten-forty2) <sup>2</sup>, [Thomas Günther](https://github.com/halbmy)<sup>3</sup>, [Andrea Balza](https://github.com/andieie)<sup>1</sup>

> <sup>1</sup> RWTH Aachen University, Applied Geophysics and Geothermal Energy, Aachen, Germany
> <sup>2</sup> Berlin University of Technology, Department of Applied Geophysics, Berlin, Germany
> <sup>3</sup> Leibniz Institute for Applied Geophysics, Hannover, Germany

|                       | Info                                                                                                                              |
| --------------------: | :-------------------------------------------------------------------------------------------------------------------------------- |
|                  When | Tuesday, April 26 • 17:00 - 19:00 UTC (starts at 08.00 a.m. CET)                                                                     |
|           Slack (Q&A) | [Software Underground](https://softwareunderground.org/) channel [#t22-tue-pygimli](https://app.slack.com/client/T094HBB9T/C039C3J7Y1M/thread/C01US4T522X-1650375402.703439) |
|           Live stream | https://youtu.be/2Hu4gDnRzlU               |
| pyGIMLi documentation | https://www.pygimli.org/documentation.html                                                                                        |
## About

pyGIMLi is an open-source library for modeling and inversion in geophysics. 
This tutorial is particularly suited for new users but adds up to last-years tutorial on Transform 2021 that was covering model building and synthetic modellings on the equation and application levels, and some standard inversion of synthetic and field data, plus how to use an own forward operator. This tutorial will add on to this and go into some more details about the underlying classes but will mainly focus on user-specific inversion:
- DataContainer, Mesh, matrix types, transformations, frameworks
- geostatistical regularization vs. classical smoothness types
- individual treatment of subsurface regions
- incorporation of prior knowledge 
- different kinds of joint and coupled inversion
- induced polarization modelling and spectrally constrained inversion
- outlook and overview of other packages in the pyGIMLi ecosystem (BERT, COMET, SAEM, custEM)


## Table of contents
- [Geophysical modeling & inversion with pyGIMLi II](#geophysical-modeling--inversion-with-pygimli)
  - [About](#about)
  - [Table of contents](#table-of-contents)
  - [BEFORE THE TUTORIAL](#before-the-tutorial)
  - [Setup instructions](#setup-instructions)
    - [Step 1: Prerequisites](#step-1-prerequisites)
    - [Step 2: Download material for the tutorial](#step-2-download-material-for-the-tutorial)
    - [Step 3: Install the tutorial environment](#step-3-install-the-tutorial-environment)
    - [Step 4: Start JupyterLab](#step-4-start-jupyterlab)
  - [Schedule](#schedule)

## BEFORE THE TUTORIAL

Make sure you've done these things **before the tutorial on Tuesday**:

1. Sign up for the [Software Underground Slack](https://softwareunderground.org/slack)
2. Join the channel [#t22-mon-pygimli](https://app.slack.com/client/T094HBB9T/C039C3J7Y1M) channel. This is where **all communication will
   happen** and where we will answer any question about installation and the tutorial
3. **Install the pyGIMLi conda environment** as described below.

## Setup instructions

> ### Quick setup for experienced users
>
> If you are working on Mac or Linux and have worked with conda and have git installed, you can copy & paste these lines separately. For all others, we recommend to carefully read the descriptions of individual steps below.
>
> ```bash
> git clone https://github.com/gimli-org/transform2022
> cd transform2022
> conda env create
> conda activate pg-transform2022
> python -c "import pygimli; pygimli.test(show=False, onlydoctests=True)"
> jupyter lab
> ```

To start the tutorial setup, please follow the next steps:

### Step 1: Prerequisites

There are a few things you'll need to follow the tutorial:

1. A working Python installation (Anaconda or Miniconda). For details on how to install Anaconda, we refer to: https://docs.anaconda.com/anaconda/install/
2. A modern web browser that works with JupyterLab or Jupyter Notebook (Internet explorer will not work)
3. Intermediate experience in Python programming (Python, numpy, matplotlib, jupyter)
4. Background on geophysical modeling and inversion

### Step 2: Download material for the tutorial

- Windows: [Download the course material](https://github.com/gimli-org/transform2022/archive/refs/heads/main.zip) and unzip it a folder of your choice.
- Mac/Linux: You can do the same as above, or alternatively open a terminal, navigate to a folder of your choice, and execute `git clone https://github.com/gimli-org/transform2022`.


### Step 3: Install the tutorial environment

1. Open a terminal (Linux & Mac) or the Anaconda Powershell Prompt (Windows). Navigate to the folder from step 2 (using the `cd` command) and type:

```
conda env create
```

2. Activate the environment in the terminal by typing:

```
conda activate pg-transform2022
```

3. To test if everything works correctly you can do the following:

```
python -c "import pygimli; pygimli.test(show=False, onlydoctests=True)"
```

If none of these commands gives an error, then your installation is working fine.
If you get any errors, please let us know on Slack at [#t22-mon-pygimli](https://app.slack.com/client/T094HBB9T/C039C3J7Y1M).

### Step 4: Start JupyterLab

1. **Windows users:** Make sure you set a default browser that is **not Internet Explorer**.
2. Activate the conda environment: `conda activate pg-transform`
3. Start JupyterLab: `jupyter lab`
4. Jupyter should open in your default web browser. We'll start from here in the
   tutorial and create a new notebook together.

## Schedule



%Identification
%%The class identifies itself and the LaTeX version needed
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Academic}[2020/09/06 Made By Liam Hughes]
\LoadClass[twoside, a4paper, 12pt]{article} %changed to 12pt


%%Preliminary definitions, needed by the options
\newcommand{\headlinecolor}{\normalcolor}
\RequirePackage{newtxtext}
\RequirePackage{newtxmath}
\renewcommand{\familydefault}{\rmdefault}

\RequirePackage[table,xcdraw]{xcolor}
\definecolor{slcolor}{HTML}{882B21}

%%This parts handles the options passed to the class.
\DeclareOption{onecolumn}{\OptionNotUsed}
%\DeclareOption{onecolumn}{\onecolstyletrue}
\DeclareOption{green}{\renewcommand{\headlinecolor}{\color{green}}}
\DeclareOption{red}{\renewcommand{\headlinecolor}{\color{slcolor}}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

%twocolumn

%Packages
\RequirePackage{tikz}
\RequirePackage{kantlipsum}
\RequirePackage{lipsum}
\RequirePackage{amsmath, amsfonts}
\let\Bbbk\relax
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{setspace}
\RequirePackage{url}
\RequirePackage{graphicx}
\RequirePackage{adjustbox}
\RequirePackage{wrapfig}
\RequirePackage[UKenglish, nodayofweek]{datetime}
\RequirePackage{caption}
\RequirePackage{subcaption}

%Packages
\RequirePackage[percent]{overpic}
\RequirePackage[figuresright]{rotating}
\RequirePackage{pdfpages}
\RequirePackage{pdflscape}
\RequirePackage{siunitx}
\RequirePackage{xfrac}
\RequirePackage{asymptote}
\RequirePackage{mathrsfs}
\RequirePackage{afterpage}
\RequirePackage{changepage}
\RequirePackage{needspace, blindtext, changepage, setspace, float, placeins}
\RequirePackage[symbol, bottom]{footmisc}

\RequirePackage{appendix}
\RequirePackage{emptypage}
\RequirePackage{etoolbox}
\RequirePackage{iftex}

\usepackage{booktabs}
\usepackage{ marvosym }

%Commands that are mine!
\newcommand{\vectg}[3]{$\text{\sfrac{#1}{#2}}  \text{$\langle$}#3\text{$\rangle$} $}
\newcommand{\dir}[1]{\text{$\langle$}#1\text{$\rangle$}}
\newcommand{\gantt}[1]{\hyperref[fig:Gantt]{#1}}
\newcommand{\vectbr}[3]{$\text{\sfrac{#1}{#2}}${[}$\text{#3}${]}}




%\sloppy - required if text falls off page
%\fussy - if you want to undo \sloppy

%Title modifications

%Paragraphs
\RequirePackage{indentfirst}
\setlength{\parindent}{3ex}
\setlength{\parskip}{0ex}

%Word Count
\newcommand{\quickwordcount}[1]{%
  \immediate\write18{python wordcount.py "#1.tex" "output/#1-words.tex"}%
  \IfFileExists{output/#1-words.tex}{\input{output/#1-words.tex}}{0 }%
}
%\quickwordcount{main} - name of main.tex document (extension not required)
%counts (text words, header words, caption words, #headers, #floats, #inlined formulae, #displayed formulae) 


%colourbox
\newcommand\mybar{\kern1pt\rule[-\dp\strutbox]{1pt}{\baselineskip}\kern1pt}
\RequirePackage{color}
\definecolor{lightgray}{gray}{0.85}
\newcommand\greybox[1]{%
  \par\noindent\colorbox{lightgray}{%
    \begin{minipage}{\textwidth}#1\end{minipage}%
  }%
  \vskip\baselineskip%
}

%Main Title
\title{%
\vspace{-2.2em} %Move the image up the page
\begin{center}
\includegraphics[width=0.42\textwidth]{assets/polyu.pdf}\par
\vspace{1.0em}
\begin{minipage}{0.92\textwidth}
\centering
\textbf{\fontsize{20}{26}\selectfont \myTitle}\par
\end{minipage}
\end{center}
}

\author{
\vspace{-0.8em}%
\textbf{\large \MyAuthor$^{1}$}\\[0.2em]
\small{$^{1}$\MyDepartment}
%\small{$^{2}$Supervisor,}
%\small{$^{3}$Co-Supervisor} Can add more names...
}


\RequirePackage{titling}
\pretitle{% add some rules
  \vskip 0.8em
    \begin{center}
}%, make the fonts bigger, make the title (only) bold
\posttitle{%
  \end{center}%
  %\noindent\vrule height 2.5pt width \textwidth
  %\vskip .75em plus .25em minus .25em%a increase the vertical spacing a bit, make this particular glue stretchier
}
\RequirePackage{microtype}
\newcommand{\Keywords}{Keyword1, keyword2, keyword3}
\newcommand{\myTitle}{Title}
\newcommand{\MyAuthor}{John J. Smith}
\newcommand{\MyDepartment}{Department}
\newcommand{\MyCourse}{Course Name}
\newcommand{\ID}{1234567}

\preauthor{%
    \vspace{0.6em}
    \begin{center}
    \vspace{0.0em}
}



\postauthor{%
    \end{center}%
    \vskip 0em
    \vspace{0.8em}
    \noindent
    \begin{minipage}{\textwidth}
    \small
    \begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}r@{}}
    \textnormal{\quickwordcount{\jobname} Words} & \textls[20]{Student ID:\ID}\\
    \multicolumn{2}{@{}c@{}}{\textls[35]{\textbf{Keywords:}} \Keywords}
    \end{tabular*}
    \end{minipage}
    \vspace{-0.3em}
}

%Date Settings
\predate{
\begin{center}\textls[10]{
}
\postdate{\end{center}\vspace{-0.6em}}}

%Custom Abstract
\RequirePackage{expl3,xparse,xcoffins}
\ExplSyntaxOn
\coffin_new:N \l_my_abstract_coffin %Ignore errors
\dim_zero_new:N \l_my_width_dim
\keys_define:nn { my / abstract }
  {
    width .code:n = {
      \dim_set:Nn \l_my_width_dim {#1\textwidth}
    }
  }

\NewDocumentCommand \myabstract { O {width=0.85} m }{%
  \keys_set:nn { my / abstract } { #1 }
  \SetVerticalCoffin \l_my_abstract_coffin {\l_my_width_dim} {#2}
  \renewcommand\maketitlehookd{%
  \noindent\rule{\textwidth}{1pt}
    \par\vspace{0.7em}
    \centering
    \TypesetCoffin \l_my_abstract_coffin
    \par\vspace{0.8em}
    \noindent\rule{\textwidth}{1pt}
    \par\vspace{0.5em}
  }

}
\ExplSyntaxOff


%Links and colours
\definecolor{azure}{rgb}{0.0, 0.5, 1.0}
\RequirePackage[bookmarks, hidelinks]{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=azure,
    filecolor=magenta,      
    urlcolor=azure,
    citecolor=azure,
    hypertexnames=true
}

%Margins and Columns
\RequirePackage[includeheadfoot,top=0.5in, bottom=0.5in, 
    left=1in, right=1in, headheight=26pt]{geometry} % room for two-line header



\setlength{\columnsep}{10mm} %Column spacing

%Badness
\widowpenalty10000
\clubpenalty10000

\hbadness=999999
\vbadness=999999

% \showboxbreadth=50 
% \showboxdepth=50

%Strechpars for paragraph neatness
\newenvironment{stretchpars}
{\par\setlength{\parfillskip}{0pt}}
{\par}


\newenvironment{changemargin}[2]{%
\begin{list}{}{%
\setlength{\topsep}{0pt}%
\setlength{\leftmargin}{#1}%
\setlength{\rightmargin}{#2}%
\setlength{\listparindent}{\parindent}%
\setlength{\itemindent}{\parindent}%
\setlength{\parsep}{\parskip}%
}%
\item[]}{\end{list}}


%Section Style
 \RequirePackage[compact]{titlesec}
    \titlespacing{\section}{2pt}{2ex}{2ex}
    \titlespacing{\subsection}{1.5pt}{1.5ex}{1.5ex}
    \titlespacing{\subsubsection}{1pt}{1ex}{1ex}

\RequirePackage{titlesec, blindtext, color, tipa}
\newcommand{\hsp}{\hspace{20pt}}

\usetikzlibrary{positioning}
\RequirePackage[absolute,overlay]{textpos}

%Headings and Footers
\RequirePackage{lastpage}
\RequirePackage{fancyhdr}
%Title Page Heading
\fancypagestyle{plain}{%


  \fancyfoot[C]{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
  \fancyhead[L]{}
  

 
  \fancyhead[C]{}
  \fancyhead[R]{}
  \fancyfoot[L]{}
}

%Fancy Headings
\pagestyle{fancy}
\fancyhead{}
\fancyfoot[C]{}
\fancyhead[C]{%
  \begin{tabular*}{\headwidth}{@{}l@{\extracolsep{\fill}}r@{}}
  \multicolumn{2}{@{}c@{}}{\small\MyCourse}\\[-0.05em]
  \footnotesize Page \thepage \hspace{1pt} of \hypersetup{linkcolor=black}{\pageref*{LastPage}} & \footnotesize\slshape\MyAuthor
  \end{tabular*}%
}
\renewcommand{\headrulewidth}{1pt} %0.5
\renewcommand{\footrulewidth}{0pt}


%Equation Style - Counters
%\renewcommand{\theequation}{\arabic{section}.\arabic{equation}}
%\counterwithin{figure}{section}
%\counterwithin{table}{section}



%Figures and Tables
\RequirePackage[compatibility=false,labelfont=it, singlelinecheck=false]{caption} %,textfont={bf,it}
\captionsetup{labelfont=bf} %textfont=bf
\renewcommand{\figurename}{Fig.}
\captionsetup[table]{labelformat=simple, labelsep=newline}
\captionsetup[figure]{labelformat=simple, labelsep=period}


%Use code inline
\RequirePackage{listings}

\ifPDFTeX
  \RequirePackage[utf8]{inputenc}
  \DeclareUnicodeCharacter{394}{$\Delta$}
  \DeclareUnicodeCharacter{3008}{$\langle$}
  \DeclareUnicodeCharacter{3009}{$\rangle$}
\fi


\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}


%References 
\RequirePackage[style=nature, maxbibnames=99, doi=false,url=false,isbn=false,hyperref=true, backref=false, natbib=true, labelnumber]{biblatex}

%Add reference files!!!
\addbibresource{references.bib}

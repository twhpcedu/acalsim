# Git in Zsh: https://git-scm.com/book/en/v2/Appendix-A%3A-Git-in-Other-Environments-Git-in-Zsh
autoload -Uz vcs_info
precmd() { vcs_info; }
zstyle ':vcs_info:git:*' formats '(%b) '
setopt PROMPT_SUBST

# Default PS1: '%n@%m %1~ %# '
PS1=$'%F{46}%n@%m%f %F{87}%~%f %F{203}${vcs_info_msg_0_}%f$ '

export LC_ALL=en_US.UTF-8

alias python=python3
alias pip=pip3

bindkey "^[[H" beginning-of-line
bindkey "^[[F" end-of-line

# Environment Variables

# Reference
# Color example: https://www.lihaoyi.com/post/Ansi/Rainbow256.png

clear

bash /docker/login.sh

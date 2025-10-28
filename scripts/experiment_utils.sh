set -e

experiment_alloc_devices() {
	index=$1; total=$2; step=$3; base=$4;
	dev_begin=$(python -c "print($index % $total + $base)")
	dev_end=$(python -c "print(($index + $step - 1) % $total + $base)")
	let "index += $step"
	if [ $dev_end -lt $dev_begin ]; then
		echo ""
	fi
	echo $(seq -s ',' $dev_begin $dev_end)
}

experiment_sanitize() {
	echo $1 |
		tr '/' '_' |   # translate / → _
		tr -d ' '  |   # delete spaces
		tr -s '_'      # squeeze repeated _
}

experiment_help() {
	cat >&2 <<EOF
	* To kill all sessions by PIDs:    pkill -f <command pattern>
	* To inspect a session:            tmux capture-pane -pt <session>
	* To list all sessions:            tmux list-sessions -F '#S' -f "#{m:<session prefix>*,#S}"
	* Pipe above to kill all sessions: | xargs -n 1 tmux kill-session -t
EOF
}

experiment_argparse() {
	argument=$1
	default_val=$2
	shift 2
	while [ $# -gt 0 ]; do
		case "$1" in
			$argument)
				echo "[experiment_argparse]: $argument $2" >&2
				echo $2
				return
				;;
			--help)
				experiment_help
				return 1
				;;
			--*)
				if [ $# -lt 2 ]; then
					echo "Argument missing value for $1" >&2
					return 1
				fi
				shift 2
				;;
			*)
				echo "Unexpected argument: $1" >&2
				return 1
		esac
	done
	echo "[experiment_argparse]: $argument $default_val" >&2
	echo $default_val
}

experiment_session() {
	session_id=exp_${1-experiment}; shift 1
	if ! tmux has-session -t $session_id; then
		tmux new-session -c `pwd` -s $session_id -d
	fi
	tput bold setaf 3; echo -n "[$session_id] "; tput sgr0
	tput bold setaf 4; echo $@; tput sgr0
	tmux send-keys -t $session_id "$@" Enter
}

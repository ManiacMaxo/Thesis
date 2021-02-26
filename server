#!/bin/bash

case $1 in
	start)
		conda run -n tf jupyter notebook --ip=0.0.0.0 --no-browser --autoreload &!
		;;
	stop)
		kill $(ps aux | grep jupyter | awk '{print $2}')
		;;
	*)
		echo "pass \"start\" or \"stop\""
		;;
esac

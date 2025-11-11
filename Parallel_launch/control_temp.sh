while true; do
    sensors | awk -v date="$(date +%Y-%m-%d_%H:%M:%S)" '
        BEGIN {output = date}
        /Package id 0:/ {output = output "," $4}
        /Core [0-9]+:/ {output = output "," $3}
        END {print output}' >> temperature_log.txt
    sleep 10 
done

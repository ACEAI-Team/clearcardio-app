sudo setcap 'cap_net_raw,cap_net_admin+eip' ./main.py
sudo setcap 'cap_net_raw,cap_net_admin+eip' ./blue.py
sudo systemctl restart bluetooth

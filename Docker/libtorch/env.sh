export PROXY_IP=$1
export PROXY_PORT=$2

echo "PROXY_ID: ${PROXY_IP}; PROXY_PORT: ${PROXY_PORT}"

export HTTP_PROXY=http://${PROXY_IP}:${PROXY_PORT}/
export HTTPS_PROXY=http://${PROXY_IP}:${PROXY_PORT}/
export FTP_PROXY=http://${PROXY_IP}:${PROXY_PORT}/
export ALL_PROXY=socks://${PROXY_IP}:${PROXY_PORT}/
export NO_PROXY=localhost,127.0.0.0/8,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export ftp_proxy=$FTP_PROXY
export all_proxy=$ALL_PROXY
export no_proxy=$NO_PROXY

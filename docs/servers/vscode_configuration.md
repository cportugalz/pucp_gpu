# Visual Studio Configuration
#### Author: Christian Portugal - <cportugal@unsa.edu.pe>

Here we detail the configuration of Visual Studio Code to work with the configuration of Servers of AI PUCP Hardware, the AI PUCP servers are configured with a proxy server knows as enterprise, after we logged in into the enterprise we can make an ssh session into discovery server, so we have to use a proxy command to facilitate this procedure.
1. Install [Visual Studio Code.](https://code.visualstudio.com/download)
2. Search Remote - SSH on extensions from VSCode and install it.
3. Click button on bottom left corner of VSCode, then **Open ssh configuration file**, and copy this: 
```bash
# We will set a 1 minute keep alive to keep the connection
# active if there is no activity to avoid unwanted disconnects
Host *
  ServerAliveInterval 60

# Specify our intermediate jump host, nothing fancy here
# we just tell what the host name is for now.
Host enterprise
  HostName 200.16.4.64
  User yourusername
# Now we will specify the actual remote host with
# the jump host as the proxy. Specify remote hostname
# as the jump-host would see it since we will be connecting
# from the jump host.
Host discovery
  HostName 172.19.4.251
  User yourusername
  ProxyCommand ssh -W %h:%p enterprise
```
only change the word 'yourusername' by your current user. 

4. Again click on the button on bottom left and then select **Connect to host** and put your password for enterprise and then for discovery server.

Have a nice coding!.
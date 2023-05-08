# Some Considerations about A Social App Server

Requirements:
* Users might login from Google, Facebook, WeChat, AliPay that needs OAuth2 authentication and authorization for resource access
* Users can transmit messages between various end devices such as phones and pads.
* Users might transmit files to other clients/devices
* User data should be encrypted

## OAuth2 Considerations for Login

For instance, below is a WeChat plugin (mini program) general working mechanism.

When a user opens the plugin app from their devices (frontend),
`wx.login({...})` is automatically invoked requesting from WeChat server for *credential code*.
WeChat server replies `res.code`.

Having successfully obtained credential code `res.code`, the plugin frontend app makes a request by `wx.request({...})` to the app backend server `'https://app.example.com/onLogin'`.

The app backend server (called *client* in the OAuth2 concept scope) receives `res.code`, then makes the below request to WeChat server (`<APPID>`, `<SECRET>` are obtained when the app is registered in WeChat as an OAuth client, and `<JSCODE>` which is the plugin frontend login response code `res.code`).
```
GET https://api.weixin.qq.com/sns/jscode2session?appid=<APPID>&secret=<SECRET>&js_code=<JSCODE>&grant_type=authorization_code
```

WeChat server (called *user data holder* in the OAuth2 concept scope) replies to the app server/client with `openid` (The unique identifier of the user) and `session_key` (The session key validates the user is login in WeChat).
App server should store the keys.
App server can then identify this user and run corresponding business logic dedicated to this user.

```javascript
wx.login({
  success (res) {
    if (res.code) {
      //Initiate Network Request
      wx.request({
        url: 'https://app.example.com/onLogin',
        data: {
          code: res.code
        },
        success(res) {
            if (res.statusCode == 200){
                // business logic such as store keys to the frontend
                globalData.openid = res.data.openid;
                globalData.session_key = res.data.session_key;
            }
        }
      })
    } else {
      console.log('Login failed! ' + res.errMsg)
    }
  }
})
```

* Client resource access: can be used to access WeChat client info such as daily active users (DAU).
This is not for user resource access (user info such as username and avatar).
```
GET https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=<APPID>&secret=<APPSECRET>
```

* User resource access: In WeChat, user related resources (only available at the moment:  `string nickName` and `string avatarUrl`) are managed by WeChat itself (no authorization/access_token required) by `wx.getUserProfile({...})`.
In standard OAuth protocol, client app should ask for `authorization_code` and `access_token` for such info.

```javascript
wx.getUserProfile({
  desc: 'Complete membership info',
  success: (res) => {
    this.setData({
      userInfo: res.userInfo,
      hasUserInfo: true
    })
  }
})
```

P.S., in WeChat current specifications, user profile should be stored in client database, so that `wx.getUserProfile({...})` is invoked only once.
Every time `wx.getUserProfile({...})` is called, frontend app would display a popup asking if user would grant authorization to the app obtaining user info. This should not be repeated invoked.

## Message Transmission and Websocket

WeChat has websocket API that maintains the communication channel until user closes the app.

```javascript
SocketTask.send({
    data: someStringMessage,
    success (res) {
        console.log("data sent");
    }
})
```

### Websocket Server

```cpp
class WebsocketServer
{
public:
    WebsocketServer();
    virtual ~WebsocketServer();

private:
    queue<thread> threadPool;    

    void run(); // a while loop constantly checking epoll

protected:
    void connectClient(ClientConnection& conn) ; // low level implementation of connection, such as by socket apis
    void disconnectClient(ClientConnection& conn) ; // low level implementation of disconnection, such as by socket apis

    void doTLSHandshake(ClientConnection& conn); // typical TLS handshake

    void handleMessageRequest();
    void handleFileRequest();
    void onMessageRequest();
    void onFileRequest();

    unordered_map<long, string> appClientOpenIdMap;
    unordered_map<long, string> appClientAuthorizationCodeMap;
    unordered_map<long, string> appClientAccessTokenMap;


};

class AppWebsocketServer : public WebsocketServer
{
public:
    AppWebsocketServer();
    virtual ~AppWebsocketServer();

    void connectAppClient(); // associate this client connection with an id
    void disconnectAppClient();
		void sendAppMessage(long clientConnIdx, const string& message);
		void recvAppMessage(long clientConnIdx, const string& message);
    void sendAppFile(long clientConnIdx, const string<uchar>& fileBlob);
		void recvAppFile(long clientConnIdx, const string<uchar>& fileBlob);

private:
    unordered_map<long, ClientConnection> appClientConnMap;
    unordered_map<long, string> appClientAuthorizationCodeUrl;
    unordered_map<long, string> appClientAccessTokenUrl;
    unordered_map<long, string> appClientAuthenticationUrl;
};
```

## File Transmission

WeChat file upload API is shown as below.

```javascript
wx.chooseImage({
  success (res) {
    const tempFilePaths = res.tempFilePaths
    wx.uploadFile({
      url: 'https://app.example.com/upload',
      filePath: tempFilePaths[0],
      name: 'file',
      formData: {
        'user': 'test'
      },
      success (res){
        const data = res.data
        //do something
      }
    })
  }
})
```
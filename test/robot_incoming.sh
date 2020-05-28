curl 'https://oapi.dingtalk.com/robot/send?access_token=a3cd7e35c31f149248a46053f51b11ad843cc50a975730e565cb3f0292f8e56b' \
   -H 'Content-Type: application/json' \
   -d '{
     "msgtype": "markdown",
     "markdown": {
         "title":"Job Info",
         "text": "Job Info

Job **127915** is finished in **121.192.191.52**!

> Server ip: **121.192.191.52**
>
> Job id: **127915**
>
> Job name: **test_notification**
>
> working directory: **/data/ch2_102/test_lsf_sub**"
     }
 }'
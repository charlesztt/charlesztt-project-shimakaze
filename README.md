# charlesztt-project-shimakaze
Project Shimakaze is a bunch of python libraries that I write and wrap up for more convenient use.

## Modules

### telegram_bot
This module is to program the Telegram bot.

#### Usage
    from shimakaze.telegram_bot.passive_bot import PassiveBot
    pb = PassiveBot()
    pb.send_message("A message")

#### Config file requirement
#####Section Name
telegram_bot

#####Options
_bot_token_: The bot token which is provided by Telegram.

_chat_id_: The chat id assigned by Telegram.

### Vision

#### mnist

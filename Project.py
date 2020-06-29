import telebot as tb
from telebot import types

import torch

import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
token = 'DLS_21_94D655BACD6A2EC0'

from backend import Style_transfer
from Images import Preproc, Normalization
from Gan import GanModel, GanScrypt

TOKEN = "1150568558:AAEFFtvnpCI0MKw9IHQbNPoDSVDQjrW5LmU"
bot = tb.TeleBot(TOKEN)


@bot.message_handler(commands=['help'])
def help_func(message):
    bot.reply_to(message, "Напечатайте \"start\", чтобы выбрать способ обработки фото")


@bot.message_handler(commands=['start'])
def start_func(message):
    bot.send_message(message.chat.id, "Загрузите редактируемую картинку")
    bot.register_next_step_handler(message, get_content)


def get_content(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("images/content.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.chat.id, "[*] Contet photo added!")
        choose_style_type(message)
    except Exception as ex:
        bot.send_message(message.chat.id, "[!] Error - {}".format(str(ex)))


def choose_style_type(message):
    markup = types.InlineKeyboardMarkup()
    one = types.InlineKeyboardButton('1 style NST', callback_data='1')
    two = types.InlineKeyboardButton('2 styles NST', callback_data='2')
    three = types.InlineKeyboardButton('Monet Gan ST', callback_data='3')
    four = types.InlineKeyboardButton('Hourse to Zebra GAN', callback_data='4')
    five = types.InlineKeyboardButton('Vangogh Gan ST', callback_data='5')
    markup.add(one, two, three, four, five)
    bot.send_message(message.chat.id, "Выберите способ обработки фото:", reply_markup=markup)


@bot.callback_query_handler(func=lambda call: True)
def query_handler(call):
    if call.data == '1':
        bot.send_message(call.message.chat.id, "Загрузите желаемый стиль")
        bot.register_next_step_handler(call.message, get_style)

    elif call.data == '2':
        bot.send_message(call.message.chat.id, "Загрузите первый стиль")
        bot.register_next_step_handler(call.message, get_style_1)
    elif call.data == '3':
        run_GAN(call.message, 'monet')
    elif call.data == '4':
        run_GAN(call.message, 'horse2zebra')
    elif call.data == '5':
        run_GAN(call.message, 'vangogh')


def get_style(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("images/style.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.chat.id, "[*] Got one style!")
        train_NST(message=message, mode=1)
    except Exception as ex:
        bot.send_message(message.chat.id, "[!] Error - {}".format(str(ex)))


def get_style_1(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("images/style_1.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.chat.id, "[*] Got first style!")
        bot.send_message(message.chat.id, "Загрузите второй стиль")
        bot.register_next_step_handler(message, get_style_2)
    except Exception as ex:
        bot.send_message(message.chat.id, "[!] Error - {}".format(str(ex)))


def get_style_2(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("images/style_2.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.chat.id, "[*] Got second style!")
        train_NST(message=message, mode=2)
    except Exception as ex:
        bot.send_message(message.chat.id, "[!] Error - {}".format(str(ex)))


# @bot.message_handler(func=lambda message: message.text == 'go')
def train_NST(message, mode):
    bot.send_message(message.chat.id, "Подождите немного! Алгоритм медленный:(")
    if mode == 1:
        style1_img = Preproc(255).image_loader('images/style.jpg')
        content_img = Preproc(255).image_loader('images/content.jpg')
        input_img = content_img.clone()
        output = Style_transfer(content_img=content_img, style1_img=style1_img,
                                style2_img=torch.zeros(1, 3, 255, 255).to(device), input_img=input_img,
                                num_steps=500, style_weight_1=100000, style_weight_2=100000, mode=1).forward()
    else:
        style1_img, style2_img, content_img = Preproc(255).transform_images("images/content.jpg",
                                                                            "images/style_1.jpg",
                                                                            "images/style_2.jpg")
        input_img = content_img.clone()
        output = Style_transfer(content_img=content_img, style1_img=style1_img,
                                style2_img=style2_img, input_img=input_img,
                                num_steps=500, style_weight_1=100000, style_weight_2=100000, mode=2).forward()

    output = output.view(output.shape[1], output.shape[2], output.shape[3])
    torchvision.utils.save_image(output, 'images/output.jpg')
    #     output = transforms.ToPILImage(mode = 'RGB')(output.cpu())
    #     output.save('output.jpg')
    photo = open('images/output.jpg', 'rb')
    bot.send_photo(message.chat.id, photo)


def run_GAN(message, name):
    if name == 'monet':
        model = 'style_monet_pretrained'
    elif name == 'horse2zebra':
        model = 'horse2zebra_pretrained'
    elif name == 'vangogh':
        model = 'style_vangogh_pretrained'

    # res = GanScrypt('images/content.jpg', model, 255)
    res = GanModel(model).forward('images/content.jpg', 255)

    torchvision.utils.save_image(res, 'images/output.jpg')
    photo = open('images/output.jpg', 'rb')
    bot.send_photo(message.chat.id, photo)

def main():
    bot.enable_save_next_step_handlers(delay=2)
    bot.load_next_step_handlers()
    bot.polling()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
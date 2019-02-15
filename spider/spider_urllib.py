#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @Time    : 2019/2/15 9:55
#
# @Author  : scutjason


"""
    task: get weibo photo use urllib
"""

import re
import os
import platform
import time

import urllib
import urllib.request

from bs4 import BeautifulSoup


def _get_path(uid):
    path = {
        'Windows': 'E:/FeigeDownload/weibo/' + uid,
        'Linux': '/mnt/d/litreily/Pictures/python/sina/' + uid
    }.get(platform.system())

    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def _get_html(url, headers):
    try:
        req = urllib.request.Request(url, headers=headers)
        page = urllib.request.urlopen(req)
        html = page.read().decode('UTF-8')

    except Exception as e:
        print("get %s failed" % url)
        return None
    return html


def _capture_images(uid, headers, path):
    filter_mode = 1  # 0-all 1-original 2-pictures
    num_pages = 1
    num_blogs = 0
    num_imgs = 0

    # regular expression of imgList and img
    imglist_reg = r'href="(https://weibo.cn/mblog/picAll/.{9}\?rl=2)"'
    imglist_pattern = re.compile(imglist_reg)
    img_reg = r'src="(http://w.{2}\.sinaimg.cn/(.{6,8})/.{32,33}.(jpg|gif))"'
    img_pattern = re.compile(img_reg)

    print('start capture picture of uid:' + uid)
    while True:
        '''
            page_num 就是分页的当前页数
            filter_mode 
                filter=0 全部微博（包含纯文本微博，转载微博）
                filter=1 原创微博（包含纯文本微博）
                filter=2 图片微博（必须含有图片，包含转载）
        '''
        url = 'https://weibo.cn/%s/profile?filter=%s&page=%d' % (uid, filter_mode, num_pages)

        # 1. get html of each page url
        html = _get_html(url, headers)

        # 2. parse the html and find all the imgList Url of each page
        soup = BeautifulSoup(html, "html.parser")
        # <div class="c" id="M_G4gb5pY8t"><div>
        blogs = soup.body.find_all(attrs={'id': re.compile(r'^M_')}, recursive=False)
        num_blogs += len(blogs)

        imgurls = []
        for blog in blogs:
            blog = str(blog)
            imglist_url = imglist_pattern.findall(blog)
            if not imglist_url:
                # 2.1 get img-url from blog that have only one pic
                imgurls += img_pattern.findall(blog)
            else:
                # 2.2 get img-urls from blog that have group pics
                html = _get_html(imglist_url[0], headers)
                imgurls += img_pattern.findall(html)

        if not imgurls:
            print('capture complete!')
            print('captured pages:%d, blogs:%d, imgs:%d' % (num_pages, num_blogs, num_imgs))
            print('directory:' + path)
            break

        # 3. download all the imgs from each imgList
        print('PAGE %d with %d images' % (num_pages, len(imgurls)))
        for img in imgurls:
            imgurl = img[0].replace(img[1], 'large')        # large表示下载原图，而不是缩略图
            num_imgs += 1
            try:
                urllib.request.urlretrieve(imgurl, '{}/{}.{}'.format(path, num_imgs, img[2]))
                # display the raw url of images
                print('\t%d\t%s' % (num_imgs, imgurl))
            except Exception as e:
                print(str(e))
                print('\t%d\t%s failed' % (num_imgs, imgurl))
            # 防止反扒
            time.sleep(300)
        num_pages += 1
        print('')


def main():
    # uids = ['5899780258','5951386664','6337948455']
    # 5899780258 jie_pai_mei_nv_jiao_du
    # 5951386664 mo_jing_jie_pai
    # 6337948455 ming_xing_tui
    # 5970428563 shou_ji_jie_pai_mei
    # 6709243169 jiao_qing_sao_zi_yang_yang
    # 6697011751 yu_xian_yu_di

    uid = '5899780258'  # myuid
    path = _get_path(uid)

    # cookie is form the above url->network->request headers
    '''
        怎么获取cookie 
        打开火狐浏览器，f12，输入 m.weibo.cn/ 选择 网络 - 消息头 - Cookie
        微博手机端的js内容比较好获取，所以用m.weibo.cn
    '''
    cookies = 'SSOLoginState=1550203179; ALF=1552795178; SCF=AnS5FnwD-YU4HmOk4uwn-5YbgGZ4tdv8IxYH8kU7sbnj2Aj1Qu0a-5kTh8S0GNtwe5aphEdQ7PdGdu1x93rNplM.; SUB=_2A25xYkl7DeRhGeNK41AS-S_Iyz-IHXVSrVczrDV6PUNbktANLRCgkW1NSQnnbzQp1qKVqmsMsSH3eVBtqsYCIj4r; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWjVafAC3Hvsa-dYziYb1ec5JpX5KMhUgL.Fo-X1hz01K2Xehe2dJLoIpRLxKML1KBLBKnLxKqL1hnLBoMLxK-L1K5L12BR1h-t; SUHB=0M2VrhqJba_mNL; _T_WM=a22f943956a0f6559bc91649c1275f31; MLOGIN=1; XSRF-TOKEN=b14dda; WEIBOCN_FROM=1110006030; M_WEIBOCN_PARAMS=uicode%3D20000174'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
        'Cookie': cookies}

    # capture imgs from sina
    _capture_images(uid, headers, path)


if __name__ == '__main__':
    main()


import argparse
import pickle
#
u0_lists = pickle.load(open('../dataset/ele2013/t/1-4_user_rid.pkl', 'rb'))
u1_lists = pickle.load(open('../dataset/ele2013/t/2-5_user_rid.pkl', 'rb'))
# #
history_u_lists_1 = pickle.load(open('../dataset/ele2013/t/1_user_rid.pkl', 'rb'))
history_ura_lists_1 = pickle.load(open('../dataset/ele2013/t/1_user_rid_ra.pkl', 'rb'))
history_ure_lists_1 = pickle.load(open('../dataset/ele2013/t/1_u_text.pkl', 'rb'))

history_u_lists_2 = pickle.load(open('../dataset/ele2013/t/2_user_rid.pkl', 'rb'))
history_ura_lists_2 = pickle.load(open('../dataset/ele2013/t/2_user_rid_ra.pkl', 'rb'))
history_ure_lists_2 = pickle.load(open('../dataset/ele2013/t/2_u_text.pkl', 'rb'))

history_u_lists_3 = pickle.load(open('../dataset/ele2013/t/3_user_rid.pkl', 'rb'))
history_ura_lists_3 = pickle.load(open('../dataset/ele2013/t/3_user_rid_ra.pkl', 'rb'))
history_ure_lists_3 = pickle.load(open('../dataset/ele2013/t/3_u_text.pkl', 'rb'))

history_u_lists_4 = pickle.load(open('../dataset/ele2013/t/4_user_rid.pkl', 'rb'))
history_ura_lists_4 = pickle.load(open('../dataset/ele2013/t/4_user_rid_ra.pkl', 'rb'))
history_ure_lists_4 = pickle.load(open('../dataset/ele2013/t/4_u_text.pkl', 'rb'))

history_u_lists_5 = pickle.load(open('../dataset/ele2013/t/5_user_rid.pkl', 'rb'))
history_ura_lists_5 = pickle.load(open('../dataset/ele2013/t/5_user_rid_ra.pkl', 'rb'))
history_ure_lists_5 = pickle.load(open('../dataset/ele2013/t/5_u_text.pkl', 'rb'))

history_u_lists_6 = pickle.load(open('../dataset/ele2013/t/6_user_rid.pkl', 'rb'))
history_ura_lists_6 = pickle.load(open('../dataset/ele2013/t/6_user_rid_ra.pkl', 'rb'))
history_ure_lists_6 = pickle.load(open('../dataset/ele2013/t/6_u_text.pkl', 'rb'))

history_v_lists = pickle.load(open('../dataset/ele2013/t/item_rid.pkl', 'rb'))
history_vra_lists = pickle.load(open('../dataset/ele2013/t/item_rid_ra.pkl', 'rb'))
history_vre_lists = pickle.load(open('../dataset/ele2013/t/i_text.pkl', 'rb'))

history_v_filter = pickle.load(open('../dataset/ele2013/t/filter_item_rid.pkl', 'rb'))
#
# u0_lists = pickle.load(open('../dataset/yelp/t/1-4_user_rid.pkl', 'rb'))
# u1_lists = pickle.load(open('../dataset/yelp/t/2-5_user_rid.pkl', 'rb'))
# #
# history_u_lists_1 = pickle.load(open('../dataset/yelp/t/user_rid_1.pkl', 'rb'))
# history_ura_lists_1 = pickle.load(open('../dataset/yelp/t/user_rid_ra_1.pkl', 'rb'))
# history_ure_lists_1 = pickle.load(open('../dataset/yelp/t/u_text_1.pkl', 'rb'))
#
# history_u_lists_2 = pickle.load(open('../dataset/yelp/t/user_rid_2.pkl', 'rb'))
# history_ura_lists_2 = pickle.load(open('../dataset/yelp/t/user_rid_ra_2.pkl', 'rb'))
# history_ure_lists_2 = pickle.load(open('../dataset/yelp/t/u_text_2.pkl', 'rb'))
#
# history_u_lists_3 = pickle.load(open('../dataset/yelp/t/user_rid_3.pkl', 'rb'))
# history_ura_lists_3 = pickle.load(open('../dataset/yelp/t/user_rid_ra_3.pkl', 'rb'))
# history_ure_lists_3 = pickle.load(open('../dataset/yelp/t/u_text_3.pkl', 'rb'))
#
# history_u_lists_4 = pickle.load(open('../dataset/yelp/t/user_rid_4.pkl', 'rb'))
# history_ura_lists_4 = pickle.load(open('../dataset/yelp/t/user_rid_ra_4.pkl', 'rb'))
# history_ure_lists_4 = pickle.load(open('../dataset/yelp/t/u_text_4.pkl', 'rb'))
#
# history_u_lists_5 = pickle.load(open('../dataset/yelp/t/user_rid_5.pkl', 'rb'))
# history_ura_lists_5 = pickle.load(open('../dataset/yelp/t/user_rid_ra_5.pkl', 'rb'))
# history_ure_lists_5 = pickle.load(open('../dataset/yelp/t/u_text_5.pkl', 'rb'))
#
# history_u_lists_6 = pickle.load(open('../dataset/yelp/t/user_rid_6.pkl', 'rb'))
# history_ura_lists_6 = pickle.load(open('../dataset/yelp/t/user_rid_ra_6.pkl', 'rb'))
# history_ure_lists_6 = pickle.load(open('../dataset/yelp/t/u_text_6.pkl', 'rb'))
#
# history_v_lists = pickle.load(open('../dataset/yelp/t/item_rid.pkl', 'rb'))
# history_vra_lists = pickle.load(open('../dataset/yelp/t/item_rid_ra.pkl', 'rb'))
# history_vre_lists = pickle.load(open('../dataset/yelp/t/i_text.pkl', 'rb'))
#
# history_v_filter = pickle.load(open('../dataset/yelp/t/filter_item_rid_y.pkl', 'rb'))


def get_data():
    data = argparse.ArgumentParser(description='model data')
    #
    data.add_argument('--filter_v_lists', default=history_v_filter)
    data.add_argument('--h_u0_lists', default=u0_lists)
    data.add_argument('--h_u1_lists', default=u1_lists)
    data.add_argument('--h_u_lists_1', default=history_u_lists_1)
    data.add_argument('--h_ura_lists_1', default=history_ura_lists_1)
    data.add_argument('--h_ure_lists_1', default=history_ure_lists_1)
    data.add_argument('--h_u_lists_2', default=history_u_lists_2)
    data.add_argument('--h_ura_lists_2', default=history_ura_lists_2)
    data.add_argument('--h_ure_lists_2', default=history_ure_lists_2)
    data.add_argument('--h_u_lists_3', default=history_u_lists_3)
    data.add_argument('--h_ura_lists_3', default=history_ura_lists_3)
    data.add_argument('--h_ure_lists_3', default=history_ure_lists_3)
    data.add_argument('--h_u_lists_4', default=history_u_lists_4)
    data.add_argument('--h_ura_lists_4', default=history_ura_lists_4)
    data.add_argument('--h_ure_lists_4', default=history_ure_lists_4)
    data.add_argument('--h_u_lists_5', default=history_u_lists_5)
    data.add_argument('--h_ura_lists_5', default=history_ura_lists_5)
    data.add_argument('--h_ure_lists_5', default=history_ure_lists_5)
    data.add_argument('--h_u_lists_6', default=history_u_lists_6)
    data.add_argument('--h_ura_lists_6', default=history_ura_lists_6)
    data.add_argument('--h_ure_lists_6', default=history_ure_lists_6)
    data.add_argument('--h_v_lists', default=history_v_lists)
    data.add_argument('--h_vra_lists', default=history_vra_lists)
    data.add_argument('--h_vre_lists', default=history_vre_lists)

    return data

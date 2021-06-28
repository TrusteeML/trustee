import collections
import copy
import csv
import datetime
import socket
from datetime import datetime as dt
from functools import reduce

import pyshark
import rootpath


def parse_pkts_per_flow(path):
    """ Parses pcap file, dividing by flow """

    packets_per_flow = {}
    flow_start_ts = {}
    flow_stop_ts = {}
    five_tuple_count = {}
    capture = pyshark.FileCapture(path, use_json=True)
    for packet in capture:
        try:
            protocol = packet.transport_layer
            source_address = packet.ip.src
            source_port = packet[packet.transport_layer].srcport
            destination_address = packet.ip.dst
            destination_port = packet[packet.transport_layer].dstport

            # bi-derectional flow
            five_tuple = (source_address, source_port, destination_address, destination_port, protocol)
            alt_five_tuple = (destination_address, destination_port, source_address, source_port, protocol)
            if five_tuple not in five_tuple_count:
                five_tuple_count[five_tuple] = 0

            key = (source_address, source_port, destination_address,
                   destination_port, protocol, five_tuple_count[five_tuple])
            alt_key = (destination_address, destination_port, source_address,
                       source_port, protocol, five_tuple_count[five_tuple])

            if key not in packets_per_flow and alt_key not in packets_per_flow:
                packets_per_flow[key] = []
                flow_start_ts[key] = packet.sniff_timestamp

            flow_key = key if key in packets_per_flow else alt_key

            print('sniff_timestamp', packet.sniff_timestamp)
            packet_ts = datetime.datetime.strptime(packet.sniff_timestamp, '%b %d, %Y %H:%M:%S.%f000 -03')
            flow_ts = datetime.datetime.strptime(flow_start_ts[flow_key], '%b %d, %Y %H:%M:%S.%f000 -03')

            print(packet_ts, flow_ts, (packet_ts - flow_ts).total_seconds())
            if ('tcp' in packet and int(packet.tcp.flags, 16) & 1) or (packet_ts - flow_ts).total_seconds() >= 600:
                # print('flags', packet.tcp.flags)
                # print('fin', int(packet.tcp.flags, 16))
                # print('fin', int(packet.tcp.flags, 16) & 1)
                flow_stop_ts[flow_key] = packet.sniff_timestamp
                five_tuple_count[five_tuple] += 1

            packets_per_flow[flow_key].append(copy.copy(packet))
        except AttributeError as e:
            # print('Error', e)
            pass

    return packets_per_flow


def extract_features(path):
    """ Extracts select features from pcap dataset """
    packets_per_flow = parse_pkts_per_flow(path)
    # print('packets_per_flow', packets_per_flow)

    with open("{}/res/dataset/eval_new_features.csv".format(rootpath.detect()), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['tls_version', 'has_heartbeat', 'has_heartbeat_size_incompat', 'label'])

        for (flow, pkts) in packets_per_flow.items():
            print('flow', flow)
            (flow_source_address, flow_source_port, flow_destination_address,
             flow_destination_port, flow_protocol, count) = flow

            # print('num packets', len(pkts))
            tls_version = 0
            has_heartbeat = 0
            heartbeat_incompatible_size = 0
            label = 'Heartbleed' if flow_source_port == '444' or flow_destination_port == '444' else 'BENIGN'

            for i in range(len(pkts)):
                try:
                    pkt = copy.copy(pkts[i])
                    if 'tls' in pkt:
                        if tls_version == 0:
                            tls_version = str(pkt.tls.record.version)

                        if has_heartbeat == 0:
                            has_heartbeat = 1 if int(pkt.tls.record.content_type, 0) == 24 else 0  # heartbeat msg

                        if int(pkt.tls.record.content_type, 0) == 24 and heartbeat_incompatible_size == 0:
                            j = i + 1
                            next_heartbeat = None
                            while j < len(pkts) and not next_heartbeat:
                                try:
                                    next_pkt = copy.copy(pkts[j])
                                    if 'tls' in next_pkt and int(next_pkt.tls.record.content_type, 0) == 24:
                                        next_heartbeat = next_pkt
                                except AttributeError as e:
                                    print('Error in while', e)
                                finally:
                                    j += 1

                            protocol = pkt.transport_layer
                            source_address = pkt.ip.src
                            source_port = pkt[pkt.transport_layer].srcport
                            destination_address = pkt.ip.dst
                            destination_port = pkt[pkt.transport_layer].dstport

                            next_protocol = next_heartbeat.transport_layer
                            next_source_address = next_heartbeat.ip.src
                            next_source_port = next_heartbeat[next_heartbeat.transport_layer].srcport
                            next_destination_address = next_heartbeat.ip.dst
                            next_destination_port = next_heartbeat[next_heartbeat.transport_layer].dstport

                            # check if its a response heartbeat
                            if source_address == next_destination_address and source_port == next_destination_port and destination_address == next_source_address and destination_port == next_source_port:
                                print('heartbeat', (source_address, source_port,
                                                    destination_address, destination_port, protocol))
                                print('heartbeat', pkt.tls)

                                print('next heartbeat', (next_source_address, next_source_port,
                                                         next_destination_address, next_destination_port, next_protocol))
                                print('heartbeat', next_heartbeat.tls)

                                heartbeat_incompatible_size = 1 if (int(pkt.tls.record.length) <= 1.5 * int(next_heartbeat.tls.record.length)) or (
                                    int(pkt.tls.record.length) >= 1.5 * int(next_heartbeat.tls.record.length)) else 0

                        # print("TLS version", str(pkt.tls.record_version))
                        # print("TLS Packet Type", pkt.tls.record_content_type)
                        # print("TLS Record Length", pkt.tls.record_length)
                except AttributeError as e:
                    # print('Error', e)
                    pass

            writer.writerow([tls_version, has_heartbeat, heartbeat_incompatible_size, label])


def main():
    """ Main block """
    # extract_features('{}/res/dataset/CIC-IDS-2017/PCAPs/HeartBleedFlow.pcap'.format(rootpath.detect()))
    extract_features('{}/res/dataset/CIC-IDS-2017/PCAPs/Wednesday-WorkingHours-HeartBleed.pcap'.format(rootpath.detect()))


if __name__ == '__main__':
    main()

import numpy as np
import struct
from math import floor

from pypcd import pypcd

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, NavSatFix


_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

# define fields for DEK pcd
FIELDS_XYZI = [ PointField(name='x',         offset=0,  datatype=PointField.FLOAT64, count=1), 
                PointField(name='y',         offset=8,  datatype=PointField.FLOAT64, count=1), 
                PointField(name='z',         offset=16, datatype=PointField.FLOAT64, count=1), 
                PointField(name='intensity', offset=24, datatype=PointField.FLOAT32, count=1)]

FIELDS_XYZ  = [ PointField(name='x',         offset=0,  datatype=PointField.FLOAT64, count=1), 
                PointField(name='y',         offset=8,  datatype=PointField.FLOAT64, count=1), 
                PointField(name='z',         offset=16, datatype=PointField.FLOAT64, count=1)]


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'
    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length
    return fmt



#######################################################

def readpcd(fname):
    return pypcd.PointCloud.from_path(fname)

def pypcd2Pointcloud2_XYZ(header,cloud):
    fmt = _get_struct_fmt(False, FIELDS_XYZ) 
    cloud_struct = struct.Struct(fmt)
    points = cloud.pc_data
    data   = points.tobytes()

    return PointCloud2( header=header,
                        height=1,
                        width=len(points),
                        is_dense=False,
                        is_bigendian=False,
                        fields=FIELDS_XYZI,
                        point_step=cloud_struct.size,
                        row_step=cloud_struct.size * len(points),
                        data=data)

def pypcd2Pointcloud2_XYZI(header,cloud):
    fmt = _get_struct_fmt(False, FIELDS_XYZI) 
    cloud_struct = struct.Struct(fmt)
    points = cloud.pc_data
    data   = points.tobytes()

    return PointCloud2( header=header,
                        height=1,
                        width=len(points),
                        is_dense=False,
                        is_bigendian=False,
                        fields=FIELDS_XYZI,
                        point_step=cloud_struct.size,
                        row_step=cloud_struct.size * len(points),
                        data=data)

def header_from_record(sd_record):   
    header_msg = Header()
    header_msg.frame_id         = sd_record['channel']
    #header_msg.stamp._sec       = floor(sd_record['timestamp'])
    #header_msg.stamp._nanosec   = floor((sd_record['timestamp'] - floor(sd_record['timestamp']))*10**9)
    header_msg.stamp._nanosec   = int(sd_record['timestamp']*10**9)
    return header_msg

def get_lidar_pc2(nusc,current_sample,sensor):
    if sensor in current_sample['data']:
        #print('has {} sensor'.format(sensor))
        #print(current_sample['data'])
        sample_data_token = current_sample['data'][sensor]

        path_to_sample_data = nusc.get_sample_data_path(sample_data_token)
        sd_record           = nusc.get('sample_data', sample_data_token) 
        #print(sd_record)
        return pypcd2Pointcloud2_XYZI(header = header_from_record(sd_record),
                                cloud  = readpcd(path_to_sample_data))
    else:
        print('has no  {} sensor'.format(sensor))
        return None

def is_valid(ego_pose):
    if  ego_pose['translation'] == [None, None, None]:
        #print(ego_pose['translation'])
        #print(False)
        return False
    else:
        #print(ego_pose['translation'])
        #print(True)
        return True

def get_gps_nsf(nusc,current_sample):
    sensor = list(current_sample['data'].keys())[0]
    sample_data_token = current_sample['data'][sensor]
    sd_record         = nusc.get('sample_data', sample_data_token) 
    ego_pose          = nusc.get('ego_pose', sd_record['ego_pose_token']) 

    if not is_valid(ego_pose): return None, None

    stamp = int(ego_pose['timestamp']*10**9)

    gps1 = NavSatFix()
    header_msg = Header()
    header_msg.frame_id         = 'GNSSONBOARD'
    header_msg.stamp._nanosec   = stamp
    gps1.header      = header_msg
    gps1.latitude    = float(ego_pose['GNSSONBOARD']['Latitude'])
    gps1.longitude   = float(ego_pose['GNSSONBOARD']['Longitude'])

    gps2 = NavSatFix()
    header_msg = Header()
    header_msg.frame_id         = 'GNSSTRIMBAL'
    header_msg.stamp._nanosec   = stamp
    gps2.header      = header_msg
    gps2.latitude    = float(ego_pose['GNSSTRIMBAL']['Latitude'])
    gps2.longitude   = float(ego_pose['GNSSTRIMBAL']['Longitude'])


    return gps1, gps2
##########################################################
# SG version of hailo-apps-inra/hailo_apps/hailo_app_python/apps/detection.py
# adds FRC network tables to send data to RoboRio

# region imports
# Standard library imports

# Third-party imports
import ntcore

import gi

gi.require_version("Gst", "1.0")
import cv2
from numpy import array

# Local application-specific imports
import hailo
from gi.repository import Gst

from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

# Logger
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class


table = ""
hailo_logger = get_logger(__name__)
# endregion imports


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        global table
        super().__init__()

        # start network tables
        ntinst = ntcore.NetworkTableInstance.getDefault()
        team = 2342     # on robot mustmatch team number (Rio is server)
        server = True  # for testing set up as server
        if server:
            print ("Setting up NetworkTables Server\n")
            ntinst.startServer()
        else:
            print ("Setting up NetworkTables client for team {}\n".format(team))
            ntinst.startClient4("hailopi5")
            ntinst.setServerTeam(team)
            #ntinst.setServerTeam("10.10.21.19")
            ntinst.startDSClient()
            
        table = ntinst.getTable("SmartDashboard")
        self.ntpub = table.getDoubleArrayTopic("HailoData").publish
        self.xlist = list[100]
        #self.ntput = ntinst.putNumberArray('Target_Data', self.xlist)
 
# part of Hailo demo code - not relevant        
#        self.new_variable = 42  # New variable example

#    def new_function(self):  # New function example
#        return "The meaning of life is: "


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------


# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    global table
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        hailo_logger.warning("Received None buffer | frame=%s", user_data.get_count())
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    frame_idx = user_data.get_count()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    hailo_logger.debug("Frame=%s | caps fmt=%s %sx%s", frame_idx, format, width, height)
    #print all the debug data
    print("PAD frame data",frame_idx, format, width, height)
    
    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    #initialize NT buffer
    #user_data.xlist = []
    newList =[]     #list of detection data to send to NT
    zoneList=[]     #list of range zone data to send to NT
    
    zone_counts = [0,0,0,0,0,0,0,0,0]
    range1_counts = [0,0,0]
    range2_counts = [0,0,0,0,0]
    range3_counts = [0,0,0,0,0,0,0]
    range4_counts = [0,0,0,0,0,0,0]
    range5_counts = [0,0,0,0,0,0,0]
    range6_counts = [0,0,0,0,0]
    ignored = 0 

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            #string_to_print += (  #commenting this out gets rid of pinted data after range couots
            #    f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            #)
            hailo_logger.debug(
                "Frame=%s | Detection person | id=%s conf=%.2f bbox=(x=%.1f,y=%.1f,w=%.1f,h=%.1f)",
                frame_idx,
                track_id,
                confidence,
                bbox.xmin(),
                bbox.ymin(),
                bbox.width(),
                bbox.height(),
            )
            #ncomment to print the full detection data
            #print(
            #    track_id,
            #    confidence,
            #    bbox.xmin(),
            #    bbox.ymin(),
            #    bbox.width(),
            #    bbox.height(), 
            #)
            detection_count += 1
            #add to the zone count
            if bbox.ymin() > 0.75:						# this zone is ca 10" from intake
                if bbox.xmin() > 0.70:
                    range1_counts[0] += 1
                elif bbox.xmin() > 0.30:
                    range1_counts[1] += 1			# fuel in this zone is ready to intake
                else: 
                    range1_counts[2] += 1
            elif bbox.ymin() > 0.50:					# distance about 12" - 24"
                if bbox.xmin() > 0.70:
                    range2_counts[0] += 1
                elif bbox.xmin() > 0.58:
                    range2_counts[1] += 1
                elif bbox.xmin() > 0.42:
                    range2_counts[2] += 1      # 12" to 24" in line with intake 
                elif bbox.xmin() > 0.30:
                    range2_counts[3] += 1
                else:
                    range2_counts[4] += 1
            elif bbox.ymin() > 0.40:           # distance 24" to 40" 
                if bbox.xmin() > 0.70:
                    range3_counts[0] += 1
                elif bbox.xmin() > 0.60:
                    range3_counts[1] += 1
                elif bbox.xmin() > 0.55:
                    range3_counts[2] += 1
                elif bbox.xmin() > 0.45:
                    range3_counts[3] += 1   		# in line with width of intake
                elif bbox.xmin() > 0.40:
                    range3_counts[4] += 1
                elif bbox.xmin() > 0.30:
                    range3_counts[5] += 1
                else:
                    range3_counts[6] += 1
            elif bbox.ymin() > 0.35:				# distance 40" - 60"
                if bbox.xmin() > 0.70:
                    range4_counts[0] += 1
                elif bbox.xmin() > 0.60:
                    range4_counts[1] += 1
                elif bbox.xmin() > 0.53:
                    range4_counts[2] += 1
                elif bbox.xmin() > 0.47:
                    range4_counts[3] += 1   # center zone ca width of intake
                elif bbox.xmin() > 0.40:
                    range4_counts[4] += 1
                elif bbox.xmin() > 0.30:
                    range4_counts[5] += 1
                else:
                    range4_counts[6] += 1
            elif bbox.ymin() > 0.30:				# distance now  60" - 100"
                if bbox.xmin() > 0.70:
                    range5_counts[0] += 1
                elif bbox.xmin() > 0.60:
                    range5_counts[1] += 1
                elif bbox.xmin() > 0.53:
                    range5_counts[2] += 1
                elif bbox.xmin() > 0.47:
                    range5_counts[3] += 1   	# in line with intake slightly wider
                elif bbox.xmin() > 0.40:
                    range5_counts[4] += 1
                elif bbox.xmin() > 0.30:
                    range5_counts[5] += 1
                else:
                    range5_counts[6] += 1
            elif bbox.ymin() > 0.25:						# distance now 100" - 160"
                if bbox.xmin() > 0.70:
                    range6_counts[0] += 1
                elif bbox.xmin() > 0.58:
                    range6_counts[1] += 1
                elif bbox.xmin() > 0.42:
                    range6_counts[2] += 1			# in line with intake wider than intake
                elif bbox.xmin() > 0.30:
                    range6_counts[3] += 1
                else:
                    range6_counts[4] += 1
            else:														# ignore everything above ymin = 0.25
             		ignored += 1									# anything in this zone is either not on floor or too far
            
            
            #create the NT record uncomment to send detection data to NT           
            #newList.append(f"[{frame_idx},{track_id},{confidence},{bbox.xmin()},{bbox.ymin()},{bbox.width()},{bbox.height()}]")
            
    print ("Detection Count = ", detection_count)
    print ("range1 counts ", range1_counts)
    print ("range2 counts ", range2_counts)
    print ("range3 counts ", range3_counts)
    print ("range4 counts ", range4_counts)
    print ("range5 counts ", range5_counts)
    print ("range6 counts ", range6_counts)
    print ("ignored counts", ignored)
    
    #uncomment line below to send the full detection data to NT
    #table.putStringArray("objects",newList)  #this sends the detection data to NT
    
    #create range zone data as a list fo strings to send to NT
    zoneList.append(f"[{detection_count},{range1_counts},{range2_counts},{range3_counts},{range4_counts},{range5_counts},{range6_counts}]")
    
    # send the range count data to NT as arrays of numbers
    table.putValue("Detection_count", detection_count)
    table.putNumberArray("Range1_counts",range1_counts)
    table.putNumberArray("Range2_counts",range2_counts)
    table.putNumberArray("Range3_counts",range3_counts)
    table.putNumberArray("Range4_counts",range4_counts)
    table.putNumberArray("range5_counts",range5_counts)
    table.putNumberArray("range6_counts",range6_counts)    
    
    #send the zone list to NT - this is same data as above - probably no need to send it both ways
    table.putStringArray("RangeCounts",zoneList) #sends all the range lists as array of strings to NT
    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(
            frame,
            f"{user_data.new_function()} {user_data.new_variable}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    #print(string_to_print)
    hailo_logger.info(string_to_print.strip())
    return Gst.PadProbeReturn.OK


def main():
    # Create an instance of the user app callback class
    hailo_logger.info("Starting Hailo Detection App...")
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()

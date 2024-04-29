import sys
import logging
from logger import logging

def error_message_detail(error, error_detail):
    _,_,exc_tb=error_detail.exc_info()
    error_message=f"Error occured in python script [{exc_tb.tb_frame.f_code.co_filename}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)


    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a=1/0 
    except Exception as e:
        logging.info("Division by zero error")
        raise CustomException(e, sys)